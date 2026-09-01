/*
Copyright 2025 The llm-d-inference-simference-sim Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// vLLM Prometheus implementation of EngineMetricsAdapter.
// See docs/metrics-refactor-design.md for the event-to-metric mapping.

package metrics

import (
	"context"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
)

const (
	VLLME2EReqLatencyMetricName           = "vllm:e2e_request_latency_seconds"
	VLLMReqQueueTimeMetricName            = "vllm:request_queue_time_seconds"
	VLLMReqInferenceTimeMetricName        = "vllm:request_inference_time_seconds"
	VLLMPrefillTimeMetricName             = "vllm:request_prefill_time_seconds"
	VLLMDecodeTimeMetricName              = "vllm:request_decode_time_seconds"
	VLLMTTFTMetricName                    = "vllm:time_to_first_token_seconds"
	VLLMTPOTMetricName                    = "vllm:time_per_output_token_seconds"
	VLLMReqTPOTMetricName                 = "vllm:request_time_per_output_token_seconds"
	VLLMInterTokenLatencyMetricName       = "vllm:inter_token_latency_seconds"
	VLLMMaxNumGenerationTokensMetricName  = "vllm:max_num_generation_tokens"
	VLLMGenerationTokensMetricName        = "vllm:request_generation_tokens"
	VLLMParamMaxTokensMetricName          = "vllm:request_params_max_tokens"
	VLLMPromptTokensMetricName            = "vllm:request_prompt_tokens"
	VLLMGenerationTokensTotalMetricName   = "vllm:generation_tokens_total"
	VLLMPromptTokensTotalMetricName       = "vllm:prompt_tokens_total"
	VLLMSuccessTotalMetricName            = "vllm:request_success_total"
	VLLMLoRARequestsMetricName            = "vllm:lora_requests_info"
	VLLMReqRunningMetricName              = "vllm:num_requests_running"
	VLLMReqWaitingMetricName              = "vllm:num_requests_waiting"
	VLLMKVCacheUsageMetricName            = "vllm:kv_cache_usage_perc"
	VLLMCacheConfigName                   = "vllm:cache_config_info"
	VLLMPrefixCacheHitsTotalMetricName    = "vllm:prefix_cache_hits_total"
	VLLMPrefixCacheQueriesTotalMetricName = "vllm:prefix_cache_queries_total"
)

var modelLabel = []string{api.PromLabelModelName}

// HistogramUpdate is the discriminated-union payload for the adapter's
// per-histogram-family channels. Event producers set Observe with a single
// observation; the fake-metrics applier sets Reset with a target bucket
// state that the updater reifies by unregistering and recreating the
// collector. Exactly one variant must be non-nil. Reset is not used yet;
// its handling lands with the applier implementation.
type HistogramUpdate struct {
	Observe *float64
	Reset   *HistogramReset
}

// HistogramReset carries the target state for a reset-shaped histogram
// update: unregister the current collector, recreate it, then Observe once
// per bucket according to Samples over the Buckets boundaries.
type HistogramReset struct {
	Buckets []float64
	Samples []int
}

// CounterUpdate is the discriminated-union payload for the adapter's
// per-counter-family channels. Event producers set Add with a delta to
// accumulate; the fake-metrics applier sets Reset to replace the counter
// with a specific value. Exactly one variant must be non-nil. Reset is not
// used yet.
type CounterUpdate struct {
	Add   *float64
	Reset *CounterReset
}

// CounterReset carries the target state for a reset-shaped counter update:
// unregister the current collector, recreate it, then Add Value.
type CounterReset struct {
	Value float64
}

// PrefixCacheStatUpdate is the discriminated-union payload for
// prefixCacheStatsChan. An event-side producer sets Query; the fake-metrics
// applier sets ResetQueries or ResetHits (a pointer to the new absolute
// value). Exactly one field is non-nil per message.
type PrefixCacheStatUpdate struct {
	Query        *PrefixCacheQueried
	ResetQueries *int64
	ResetHits    *int64
}

// RequestSuccessUpdate is the discriminated-union payload for
// requestSuccessChan. Event-side producers set Success; the fake-metrics
// applier sets exactly one of the *Reset fields.
type RequestSuccessUpdate struct {
	Success              *RequestSucceeded
	SuccessTotalReset    *SuccessTotalReset
	TokenMetricReset     *TokenMetricReset
	ParamsMaxTokensReset *HistogramReset
	MaxNumGenTokensReset *HistogramReset
}

// SuccessTotalReset carries the target state for the request_success_total
// counter: unregister, recreate, then Add per (reason, count) pair. A nil
// or empty Reasons map records nothing after the recreate.
type SuccessTotalReset struct {
	Reasons map[string]int64
}

// TokenMetricReset carries the target state for one leg of a token
// histogram + total counter pair. Samples == nil leaves the histogram
// untouched. ExplicitTotal overrides the histogram-derived sum; when both
// are nil the counter is left untouched too.
type TokenMetricReset struct {
	Kind          VLLMTokenMetricKind
	Buckets       []float64
	Samples       []int
	ExplicitTotal *int64
}

// LoRAUpdate is the discriminated-union payload for lorasChan. Event-side
// producers set Usage; the fake-metrics applier sets Reset.
type LoRAUpdate struct {
	Usage *loraUsage
	Reset *LoRAReset
}

// LoRAReset carries the target state for lora_requests_info: unregister,
// recreate, then stamp one series per entry (or a single zero-adapter row
// with the current timestamp when Entries is empty).
type LoRAReset struct {
	MaxLoRAs int
	Entries  []common.LorasMetrics
}

// VLLMMetricsAdapter implements EngineMetricsAdapter and produces the vLLM
// Prometheus metric surface documented in docs/metrics.md.
//
// The adapter operates as a two-stage pipeline:
//
//  1. Event drainers (spawned from Start) read BUS events - the first set
//     of channels - and dispatch to the matching On<Event> handler. Each
//     handler is a thin producer: it re-emits the event, possibly split
//     into multiple metric-scoped values, onto the second set of channels
//     (one per Prometheus metric family).
//
//  2. Per-metric family updater goroutines drain the second set and own the Prometheus writes.
type VLLMMetricsAdapter struct {
	logger   logr.Logger
	registry *prometheus.Registry
	config   common.Configuration

	// gauges
	runningRequests        *prometheus.GaugeVec
	waitingRequests        *prometheus.GaugeVec
	kvCacheUsagePercentage *prometheus.GaugeVec
	loraInfo               *prometheus.GaugeVec
	cacheConfig            *prometheus.GaugeVec

	// histograms
	ttft                    *prometheus.HistogramVec
	tpot                    *prometheus.HistogramVec
	interTokenLatency       *prometheus.HistogramVec
	reqTpot                 *prometheus.HistogramVec
	e2eReqLatency           *prometheus.HistogramVec
	reqQueueTime            *prometheus.HistogramVec
	reqInferenceTime        *prometheus.HistogramVec
	reqPrefillTime          *prometheus.HistogramVec
	reqDecodeTime           *prometheus.HistogramVec
	requestPromptTokens     *prometheus.HistogramVec
	requestGenerationTokens *prometheus.HistogramVec
	maxNumGenerationTokens  *prometheus.HistogramVec
	requestParamsMaxTokens  *prometheus.HistogramVec

	// counters
	promptTokensTotal       *prometheus.CounterVec
	generationTokensTotal   *prometheus.CounterVec
	requestSuccessTotal     *prometheus.CounterVec
	prefixCacheHitsTotal    *prometheus.CounterVec
	prefixCacheQueriesTotal *prometheus.CounterVec

	// Channels: one per Prometheus metric family. Handlers push
	// here; the updater goroutines below drain each and perform the actual
	// Prometheus mutation. Created in Start once ctx.Done() is available.
	runReqChan            common.Channel[common.MetricInfo]
	waitingReqChan        common.Channel[common.MetricInfo]
	kvCacheUsageChan      common.Channel[common.MetricInfo]
	ttftChan              common.Channel[HistogramUpdate]
	tpotChan              common.Channel[HistogramUpdate]
	interTokenLatencyChan common.Channel[HistogramUpdate]
	e2eReqLatencyChan     common.Channel[HistogramUpdate]
	reqQueueTimeChan      common.Channel[HistogramUpdate]
	reqInferenceTimeChan  common.Channel[HistogramUpdate]
	reqPrefillTimeChan    common.Channel[HistogramUpdate]
	reqDecodeTimeChan     common.Channel[HistogramUpdate]
	reqTpotChan           common.Channel[HistogramUpdate]
	prefixCacheStatsChan  common.Channel[PrefixCacheStatUpdate]
	requestSuccessChan    common.Channel[RequestSuccessUpdate]
	lorasChan             common.Channel[LoRAUpdate]

	// LoRA ref-counted sets, mutated only by lorasUpdater. sync.Map matches
	// the shape used by metrics.go so the two implementations stay in sync.
	runningLoras sync.Map
	waitingLoras sync.Map

	// nWaitingReqs / nRunningReqs are the adapter-local counters mirrored
	// onto the num_requests_{waiting,running} gauges.
	nWaitingReqs int64
	nRunningReqs int64

	// fakeController drives fake-metrics application when
	// config.FakeMetrics != nil. Nil otherwise.
	fakeController *VLLMFakeMetricsController
}

// NewVLLMMetricsAdapter fully initializes the adapter: registers every
// Prometheus collector, stamps initial values, constructs the per-metric
// channels, and spawns one updater goroutine per channel. When it returns,
// the adapter is ready to accept On<Event> calls. Call Start(ctx, bus)
// to wire the event bus in; ctx must be the same one passed here so the
// bus drainers and metric updaters share a shutdown signal.
//
// Returns an error if any collector fails to register (typically a name
// collision, which indicates a programmer error).
func NewVLLMMetricsAdapter(ctx context.Context, registry *prometheus.Registry, logger logr.Logger, config common.Configuration) (*VLLMMetricsAdapter, error) {
	m := &VLLMMetricsAdapter{
		logger:   logger,
		registry: registry,
		config:   config,
	}

	if err := m.buildMetrics(); err != nil {
		return nil, err
	}
	if err := m.createAndStartPrometheusChannels(ctx); err != nil {
		return nil, err
	}
	m.setInitialValues()

	if m.config.FakeMetrics != nil {
		m.fakeController = NewVLLMFakeMetricsController(&m.config, m, logger)
	}
	return m, nil
}

func (m *VLLMMetricsAdapter) Close() error {
	if m.fakeController != nil {
		m.fakeController.Close()
	}
	return nil
}

// Registry returns the private Prometheus registry the adapter publishes
// into. Served on the /metrics_new endpoint so the event-driven surface
// can be validated against the legacy /metrics output.
func (m *VLLMMetricsAdapter) Registry() *prometheus.Registry {
	return m.registry
}

// Start wires the event bus into the adapter by spawning one drainer
// goroutine per bus channel. Each drainer calls the matching On<Event>
// handler, which forwards the event onto the per-metric channel that
// NewVLLMMetricsAdapter already stood up.
func (m *VLLMMetricsAdapter) Start(ctx context.Context, bus *MetricsBus) error {
	go drain(ctx, bus.RequestQueued, m.OnRequestQueued)
	go drain(ctx, bus.RequestDequeued, m.OnRequestDequeued)
	go drain(ctx, bus.RequestRunning, m.OnRequestRunning)
	go drain(ctx, bus.PrefillStarted, m.OnPrefillStarted)
	go drain(ctx, bus.PrefillEnded, m.OnPrefillEnded)
	go drain(ctx, bus.DecodeStarted, m.OnDecodeStarted)
	go drain(ctx, bus.TokenGenerated, m.OnTokenGenerated)
	go drain(ctx, bus.DecodeEnded, m.OnDecodeEnded)
	go drain(ctx, bus.RequestSucceeded, m.OnRequestSucceeded)
	go drain(ctx, bus.RequestFailed, m.OnRequestFailed)
	go drain(ctx, bus.KVCacheUsage, m.OnKVCacheUsageChanged)
	go drain(ctx, bus.PrefixCacheQuery, m.OnPrefixCacheQueried)

	if m.fakeController != nil {
		if err := m.fakeController.SetInitial(m.config.FakeMetrics); err != nil {
			return err
		}
		m.fakeController.Start(ctx)
	}

	return nil
}

func (m *VLLMMetricsAdapter) createAndStartPrometheusChannels(ctx context.Context) error {
	maxNumberOfRequests := (m.config.MaxNumSeqs + m.config.MaxWaitingQueueLength) * 2
	maxNumberOfRunningRequests := m.config.MaxNumSeqs * 2
	maxNumberOfWaitingRequests := m.config.MaxWaitingQueueLength * 2

	m.runReqChan = common.Channel[common.MetricInfo]{
		Channel: make(chan common.MetricInfo, maxNumberOfRunningRequests),
		Name:    "vllm.runReqChan",
		Done:    ctx.Done(),
	}
	go m.runningRequestsUpdater(ctx)

	m.waitingReqChan = common.Channel[common.MetricInfo]{
		Channel: make(chan common.MetricInfo, maxNumberOfWaitingRequests),
		Name:    "vllm.waitingReqChan",
		Done:    ctx.Done(),
	}
	go m.waitingRequestsUpdater(ctx)

	m.kvCacheUsageChan = common.Channel[common.MetricInfo]{
		Channel: make(chan common.MetricInfo, maxNumberOfRunningRequests),
		Name:    "vllm.kvCacheUsageChan",
		Done:    ctx.Done(),
	}
	go m.kvCacheUsageUpdater(ctx)

	m.ttftChan = common.Channel[HistogramUpdate]{
		Channel: make(chan HistogramUpdate, maxNumberOfRunningRequests),
		Name:    "vllm.ttftChan",
		Done:    ctx.Done(),
	}
	go m.ttftUpdater(ctx)

	m.tpotChan = common.Channel[HistogramUpdate]{
		Channel: make(chan HistogramUpdate, maxNumberOfRunningRequests*m.config.MaxModelLen),
		Name:    "vllm.tpotChan",
		Done:    ctx.Done(),
	}
	go m.tpotUpdater(ctx)

	m.interTokenLatencyChan = common.Channel[HistogramUpdate]{
		Channel: make(chan HistogramUpdate, maxNumberOfRunningRequests*m.config.MaxModelLen),
		Name:    "vllm.interTokenLatencyChan",
		Done:    ctx.Done(),
	}
	go m.interTokenLatencyUpdater(ctx)

	m.e2eReqLatencyChan = common.Channel[HistogramUpdate]{
		Channel: make(chan HistogramUpdate, maxNumberOfRunningRequests),
		Name:    "vllm.e2eReqLatencyChan",
		Done:    ctx.Done(),
	}
	go m.e2eReqLatencyUpdater(ctx)

	m.reqQueueTimeChan = common.Channel[HistogramUpdate]{
		Channel: make(chan HistogramUpdate, maxNumberOfWaitingRequests),
		Name:    "vllm.reqQueueTimeChan",
		Done:    ctx.Done(),
	}
	go m.reqQueueTimeUpdater(ctx)

	m.reqInferenceTimeChan = common.Channel[HistogramUpdate]{
		Channel: make(chan HistogramUpdate, maxNumberOfRunningRequests),
		Name:    "vllm.reqInferenceTimeChan",
		Done:    ctx.Done(),
	}
	go m.reqInferenceTimeUpdater(ctx)

	m.reqPrefillTimeChan = common.Channel[HistogramUpdate]{
		Channel: make(chan HistogramUpdate, maxNumberOfRunningRequests),
		Name:    "vllm.reqPrefillTimeChan",
		Done:    ctx.Done(),
	}
	go m.reqPrefillTimeUpdater(ctx)

	m.reqDecodeTimeChan = common.Channel[HistogramUpdate]{
		Channel: make(chan HistogramUpdate, maxNumberOfRunningRequests),
		Name:    "vllm.reqDecodeTimeChan",
		Done:    ctx.Done(),
	}
	go m.reqDecodeTimeUpdater(ctx)

	m.reqTpotChan = common.Channel[HistogramUpdate]{
		Channel: make(chan HistogramUpdate, maxNumberOfRunningRequests),
		Name:    "vllm.reqTpotChan",
		Done:    ctx.Done(),
	}
	go m.reqTpotUpdater(ctx)

	m.prefixCacheStatsChan = common.Channel[PrefixCacheStatUpdate]{
		Channel: make(chan PrefixCacheStatUpdate, maxNumberOfRunningRequests),
		Name:    "vllm.prefixCacheStatsChan",
		Done:    ctx.Done(),
	}
	go m.prefixCacheStatsUpdater(ctx)

	m.requestSuccessChan = common.Channel[RequestSuccessUpdate]{
		Channel: make(chan RequestSuccessUpdate, maxNumberOfRunningRequests),
		Name:    "vllm.requestSuccessChan",
		Done:    ctx.Done(),
	}
	go m.recordRequestUpdater(ctx)

	m.lorasChan = common.Channel[LoRAUpdate]{
		Channel: make(chan LoRAUpdate, maxNumberOfRequests),
		Name:    "vllm.lorasChan",
		Done:    ctx.Done(),
	}
	go m.lorasUpdater(ctx)

	return nil
}

// observation wraps a single Observe value for a histogram-family channel.
func observation(v float64) HistogramUpdate {
	return HistogramUpdate{Observe: &v}
}

// applyHistogramUpdate is the switch every histogram updater runs. Observe
// records a single event-side observation; Reset unregisters the current
// collector, invokes recreate to stand up a fresh one in its place, then
// replays observations according to the target bucket state.
func (m *VLLMMetricsAdapter) applyHistogramUpdate(histPP **prometheus.HistogramVec, recreate func() error, upd HistogramUpdate) {
	switch {
	case upd.Observe != nil:
		m.reportHistogramValue(*histPP, *upd.Observe)
	case upd.Reset != nil:
		m.applyHistogramReset(histPP, recreate, upd.Reset)
	}
}

// applyHistogramReset unregisters the current histogram collector,
// recreates it via recreate, then replays observations per the reset
// target. Called only from updater goroutines, so no external
// synchronisation is required.
func (m *VLLMMetricsAdapter) applyHistogramReset(histPP **prometheus.HistogramVec, recreate func() error, reset *HistogramReset) {
	m.registry.Unregister(*histPP)
	if err := recreate(); err != nil {
		m.logger.Error(err, "failed to recreate histogram during fake-metrics reset")
		return
	}
	InitFakeHistogram(*histPP, m.config.DisplayModelName, reset.Buckets, reset.Samples)
}

// drain reads events from ch and dispatches them to fn until ctx is done.
func drain[E any](ctx context.Context, ch common.Channel[E], fn func(E)) {
	for {
		select {
		case <-ctx.Done():
			return
		case event := <-ch.Channel:
			fn(event)
		}
	}
}

// -- Event handlers (stage 1: bus -> per-metric channels) -------------------

func (m *VLLMMetricsAdapter) OnRequestReceived(_ RequestReceived) {
	// State marker; no exposed metric today.
}

func (m *VLLMMetricsAdapter) OnRequestRejected(_ RequestRejected) {
	// State marker; no exposed metric today.
}

// request queued
// - update number of waiting requests
// - update LoRA state if applicable
func (m *VLLMMetricsAdapter) OnRequestQueued(ev RequestQueued) {
	if m.config.FakeMetrics != nil {
		return
	}
	common.WriteToChannel(m.waitingReqChan,
		common.MetricInfo{Value: 1, IsFake: ev.IsFake}, m.logger)

	if ev.IsLoRA {
		common.WriteToChannel(m.lorasChan,
			LoRAUpdate{Usage: &loraUsage{name: ev.Model, state: waitingUsageState}}, m.logger)
	}
}

// request dequeued
// - update number of waiting requests
// - update queue time histogram
// lora will be marked as runnning in OnRequestRunning
func (m *VLLMMetricsAdapter) OnRequestDequeued(ev RequestDequeued) {
	if m.config.FakeMetrics != nil {
		return
	}
	common.WriteToChannel(m.waitingReqChan,
		common.MetricInfo{Value: -1, IsFake: ev.IsFake}, m.logger)

	common.WriteToChannel(m.reqQueueTimeChan, observation(ev.QueueTime), m.logger)
}

// request running
// - update number of running requests
// - update LoRA state if applicable
func (m *VLLMMetricsAdapter) OnRequestRunning(ev RequestRunning) {
	if m.config.FakeMetrics != nil {
		return
	}
	common.WriteToChannel(m.runReqChan,
		common.MetricInfo{Value: 1, IsFake: ev.IsFake}, m.logger)

	if ev.IsLoRA {
		common.WriteToChannel(m.lorasChan,
			LoRAUpdate{Usage: &loraUsage{name: ev.Model, state: runningUsageState}}, m.logger)
	}
}

// prefill started
func (m *VLLMMetricsAdapter) OnPrefillStarted(_ PrefillStarted) {
	// State marker.
}

// prefill step ended
// - update prefill time histogram
// - update TTFT histogram
func (m *VLLMMetricsAdapter) OnPrefillEnded(ev PrefillEnded) {
	if m.config.FakeMetrics != nil {
		return
	}
	common.WriteToChannel(m.reqPrefillTimeChan, observation(ev.PrefillDuration), m.logger)
	common.WriteToChannel(m.ttftChan, observation(ev.PrefillDuration), m.logger)
}

func (m *VLLMMetricsAdapter) OnDecodeStarted(_ DecodeStarted) {
	// State marker.
}

// token generated
// - update tpot and itl latency histograms
func (m *VLLMMetricsAdapter) OnTokenGenerated(ev TokenGenerated) {
	if m.config.FakeMetrics != nil {
		return
	}
	obs := observation(ev.InterTokenLatency)
	common.WriteToChannel(m.tpotChan, obs, m.logger)
	common.WriteToChannel(m.interTokenLatencyChan, obs, m.logger)
}

// decode ended
// - update decode time histogram
// - update requests tpot histogram
func (m *VLLMMetricsAdapter) OnDecodeEnded(ev DecodeEnded) {
	if m.config.FakeMetrics != nil {
		return
	}
	common.WriteToChannel(m.reqDecodeTimeChan, observation(ev.DecodeDuration), m.logger)

	if ev.GenerationTokens > 0 {
		common.WriteToChannel(m.reqTpotChan,
			observation(ev.DecodeDuration/float64(ev.GenerationTokens)), m.logger)
	}
}

// request processing finished successfully
// - update all relevant metrics
func (m *VLLMMetricsAdapter) OnRequestSucceeded(ev RequestSucceeded) {
	if m.config.FakeMetrics != nil {
		return
	}
	common.WriteToChannel(m.requestSuccessChan, RequestSuccessUpdate{Success: &ev}, m.logger)

	common.WriteToChannel(m.e2eReqLatencyChan, observation(ev.E2ELatency), m.logger)
	common.WriteToChannel(m.reqInferenceTimeChan, observation(ev.InferenceTime), m.logger)

	m.finishRunning(ev.Model, ev.IsLoRA, ev.IsFake)
}

// request processing failed
// - update all relevant metrics
func (m *VLLMMetricsAdapter) OnRequestFailed(ev RequestFailed) {
	if m.config.FakeMetrics != nil {
		return
	}
	common.WriteToChannel(m.e2eReqLatencyChan, observation(ev.E2ELatency), m.logger)
	common.WriteToChannel(m.reqInferenceTimeChan, observation(ev.InferenceTime), m.logger)

	m.finishRunning(ev.Model, ev.IsLoRA, ev.IsFake)

	if ev.Err != nil {
		m.logger.V(logging.DEBUG).Info("request failed", "model", ev.Model, "err", ev.Err.Error())
	}
}

// change in kv cache utilization
// - update kv cache usage gauge
func (m *VLLMMetricsAdapter) OnKVCacheUsageChanged(ev KVCacheUsageChanged) {
	if m.config.FakeMetrics != nil {
		return
	}
	common.WriteToChannel(m.kvCacheUsageChan,
		common.MetricInfo{Value: ev.KVCacheUsagePerc, IsFake: ev.IsFake}, m.logger)
}

// change in prefix cache utilization
// - update prefix cache hits and queries counters
func (m *VLLMMetricsAdapter) OnPrefixCacheQueried(ev PrefixCacheQueried) {
	if m.config.FakeMetrics != nil {
		return
	}
	common.WriteToChannel(m.prefixCacheStatsChan, PrefixCacheStatUpdate{Query: &ev}, m.logger)
}

// finishRunning fans a request-terminal event out to the running-counter
// channel (as a -1 delta) and, for LoRA requests, to the LoRA state channel.
func (m *VLLMMetricsAdapter) finishRunning(model string, isLoRA, isFake bool) {
	common.WriteToChannel(m.runReqChan,
		common.MetricInfo{Value: -1, IsFake: isFake}, m.logger)
	if isLoRA {
		common.WriteToChannel(m.lorasChan,
			LoRAUpdate{Usage: &loraUsage{name: model, state: doneUsageState}}, m.logger)
	}
}

// -- Updaters (per-metric channels -> Prometheus) ------------------

func (m *VLLMMetricsAdapter) waitingRequestsUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case upd := <-m.waitingReqChan.Channel:
			if (m.config.FakeMetrics != nil) != upd.IsFake {
				continue
			}
			if upd.IsFake {
				m.nWaitingReqs = int64(upd.Value)
			} else {
				m.nWaitingReqs += int64(upd.Value)
			}
			m.reportWaitingRequests()
		}
	}
}

func (m *VLLMMetricsAdapter) runningRequestsUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case upd := <-m.runReqChan.Channel:
			if (m.config.FakeMetrics != nil) != upd.IsFake {
				continue
			}
			if upd.IsFake {
				m.nRunningReqs = int64(upd.Value)
			} else {
				m.nRunningReqs += int64(upd.Value)
			}
			m.reportRunningRequests()
		}
	}
}

func (m *VLLMMetricsAdapter) kvCacheUsageUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case value := <-m.kvCacheUsageChan.Channel:
			if (m.config.FakeMetrics != nil) == value.IsFake {
				m.reportKVCacheUsage(value.Value)
			}
		}
	}
}

func (m *VLLMMetricsAdapter) prefixCacheStatsUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case upd := <-m.prefixCacheStatsChan.Channel:
			switch {
			case upd.Query != nil:
				m.reportPrefixCacheStats(*upd.Query)
			case upd.ResetQueries != nil:
				m.applyCounterReset(&m.prefixCacheQueriesTotal, m.createAndRegisterPrefixCacheQueriesTotalCounter,
					m.config.DisplayModelName, float64(*upd.ResetQueries))
			case upd.ResetHits != nil:
				m.applyCounterReset(&m.prefixCacheHitsTotal, m.createAndRegisterPrefixCacheHitsTotalCounter,
					m.config.DisplayModelName, float64(*upd.ResetHits))
			}
		}
	}
}

func (m *VLLMMetricsAdapter) ttftUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case upd := <-m.ttftChan.Channel:
			m.applyHistogramUpdate(&m.ttft, m.createAndRegisterTTFTHistogram, upd)
		}
	}
}

func (m *VLLMMetricsAdapter) tpotUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case upd := <-m.tpotChan.Channel:
			m.applyHistogramUpdate(&m.tpot, m.createAndRegisterTPOTHistogram, upd)
		}
	}
}

func (m *VLLMMetricsAdapter) interTokenLatencyUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case upd := <-m.interTokenLatencyChan.Channel:
			m.applyHistogramUpdate(&m.interTokenLatency, m.createAndRegisterInterTokenLatencyHistogram, upd)
		}
	}
}

func (m *VLLMMetricsAdapter) e2eReqLatencyUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case upd := <-m.e2eReqLatencyChan.Channel:
			m.applyHistogramUpdate(&m.e2eReqLatency, m.createAndRegisterE2EReqLatencyHistogram, upd)
		}
	}
}

func (m *VLLMMetricsAdapter) reqQueueTimeUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case upd := <-m.reqQueueTimeChan.Channel:
			m.applyHistogramUpdate(&m.reqQueueTime, m.createAndRegisterReqQueueTimeHistogram, upd)
		}
	}
}

func (m *VLLMMetricsAdapter) reqInferenceTimeUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case upd := <-m.reqInferenceTimeChan.Channel:
			m.applyHistogramUpdate(&m.reqInferenceTime, m.createAndRegisterReqInferenceTimeHistogram, upd)
		}
	}
}

func (m *VLLMMetricsAdapter) reqPrefillTimeUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case upd := <-m.reqPrefillTimeChan.Channel:
			m.applyHistogramUpdate(&m.reqPrefillTime, m.createAndRegisterReqPrefillTimeHistogram, upd)
		}
	}
}

func (m *VLLMMetricsAdapter) reqDecodeTimeUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case upd := <-m.reqDecodeTimeChan.Channel:
			m.applyHistogramUpdate(&m.reqDecodeTime, m.createAndRegisterReqDecodeTimeHistogram, upd)
		}
	}
}

func (m *VLLMMetricsAdapter) reqTpotUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case upd := <-m.reqTpotChan.Channel:
			m.applyHistogramUpdate(&m.reqTpot, m.createAndRegisterReqTpotHistogram, upd)
		}
	}
}

// lorasUpdater consumes LoRA state transitions and republishes
// lora_requests_info. Waiting and running sets are separate gauges
// projected onto the same metric via labels, so they share this goroutine.
// Reset payloads (from the fake-metrics applier) unregister the collector,
// recreate it, and stamp the supplied entries.
func (m *VLLMMetricsAdapter) lorasUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case upd := <-m.lorasChan.Channel:
			switch {
			case upd.Usage != nil:
				switch upd.Usage.state {
				case waitingUsageState:
					m.incrementLoraRefCount(upd.Usage.name, &m.waitingLoras)
				case runningUsageState:
					m.decrementLoraRefCount(upd.Usage.name, &m.waitingLoras)
					m.incrementLoraRefCount(upd.Usage.name, &m.runningLoras)
				case doneUsageState:
					m.decrementLoraRefCount(upd.Usage.name, &m.runningLoras)
				}
				m.reportLoras()
			case upd.Reset != nil:
				m.applyLoRAReset(upd.Reset)
			}
		}
	}
}

func (m *VLLMMetricsAdapter) recordRequestUpdater(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case upd := <-m.requestSuccessChan.Channel:
			switch {
			case upd.Success != nil:
				m.recordRequestMetricsOnSuccess(*upd.Success)
			case upd.SuccessTotalReset != nil:
				m.applySuccessTotalReset(upd.SuccessTotalReset.Reasons)
			case upd.TokenMetricReset != nil:
				m.applyTokenMetricReset(upd.TokenMetricReset)
			case upd.ParamsMaxTokensReset != nil:
				m.applyHistogramReset(&m.requestParamsMaxTokens,
					m.createAndRegisterReqParamsMaxTokensHistogram, upd.ParamsMaxTokensReset)
			case upd.MaxNumGenTokensReset != nil:
				m.applyHistogramReset(&m.maxNumGenerationTokens,
					m.createAndRegisterMaxNumGenerationTokensHistogram, upd.MaxNumGenTokensReset)
			}
		}
	}
}

// applyCounterReset unregisters counterPP, recreates it via recreate, then
// records the target value via a single Add. Called only from updater
// goroutines.
func (m *VLLMMetricsAdapter) applyCounterReset(counterPP **prometheus.CounterVec, recreate func() error, modelName string, value float64) {
	m.registry.Unregister(*counterPP)
	if err := recreate(); err != nil {
		m.logger.Error(err, "failed to recreate counter during fake-metrics reset")
		return
	}
	(*counterPP).WithLabelValues(modelName).Add(value)
}

// applySuccessTotalReset unregisters requestSuccessTotal, recreates it,
// then Adds each (reason, count) pair. Nil or empty reasons leaves the
// recreated counter with no series stamped.
func (m *VLLMMetricsAdapter) applySuccessTotalReset(reasons map[string]int64) {
	m.registry.Unregister(m.requestSuccessTotal)
	if err := m.createAndRegisterRequestSuccessTotalCounter(); err != nil {
		m.logger.Error(err, "failed to recreate request_success_total counter during fake-metrics reset")
		return
	}
	for reason, count := range reasons {
		m.requestSuccessTotal.WithLabelValues(m.config.DisplayModelName, reason).Add(float64(count))
	}
}

// applyTokenMetricReset applies a reset for one leg of a token histogram +
// total counter pair. Samples == nil leaves the histogram untouched.
// ExplicitTotal overrides the derived sum; when both are nil the counter
// is left untouched too.
func (m *VLLMMetricsAdapter) applyTokenMetricReset(reset *TokenMetricReset) {
	var histPP **prometheus.HistogramVec
	var counterPP **prometheus.CounterVec
	var recreateHist, recreateCounter func() error
	switch reset.Kind {
	case VLLMTokenMetricPrompt:
		histPP = &m.requestPromptTokens
		counterPP = &m.promptTokensTotal
		recreateHist = m.createAndRegisterReqPromptTokensHistogram
		recreateCounter = m.createAndRegisterPromptTokensTotalCounter
	case VLLMTokenMetricGeneration:
		histPP = &m.requestGenerationTokens
		counterPP = &m.generationTokensTotal
		recreateHist = m.createAndRegisterReqGenerationTokensHistogram
		recreateCounter = m.createAndRegisterGenerationTokensTotalCounter
	default:
		return
	}

	var histTotal *int64
	if reset.Samples != nil {
		m.registry.Unregister(*histPP)
		if err := recreateHist(); err != nil {
			m.logger.Error(err, "failed to recreate token histogram during fake-metrics reset")
			return
		}
		histTotal = InitFakeHistogram(*histPP, m.config.DisplayModelName, reset.Buckets, reset.Samples)
	}

	if reset.Samples == nil && reset.ExplicitTotal == nil {
		return
	}

	m.registry.Unregister(*counterPP)
	if err := recreateCounter(); err != nil {
		m.logger.Error(err, "failed to recreate token counter during fake-metrics reset")
		return
	}
	total := histTotal
	if reset.ExplicitTotal != nil {
		total = reset.ExplicitTotal
	}
	if total != nil {
		(*counterPP).WithLabelValues(m.config.DisplayModelName).Add(float64(*total))
	}
}

// applyLoRAReset unregisters loraInfo, recreates it, then stamps one
// series per entry. Empty entries emits a single zero-adapter row with the
// current timestamp (matching the fake-metrics default).
func (m *VLLMMetricsAdapter) applyLoRAReset(reset *LoRAReset) {
	m.registry.Unregister(m.loraInfo)
	if err := m.createAndRegisterLoraInfoGauge(); err != nil {
		m.logger.Error(err, "failed to recreate lora_requests_info gauge during fake-metrics reset")
		return
	}
	if len(reset.Entries) == 0 {
		m.loraInfo.WithLabelValues(
			strconv.Itoa(reset.MaxLoRAs),
			"",
			"",
		).Set(float64(time.Now().Unix()))
		return
	}
	for _, entry := range reset.Entries {
		m.loraInfo.WithLabelValues(
			strconv.Itoa(reset.MaxLoRAs),
			entry.RunningLoras,
			entry.WaitingLoras,
		).Set(entry.Timestamp)
	}
}

// -- Report helpers (Prometheus writes) -------------------------------------

func (m *VLLMMetricsAdapter) reportHistogramValue(hist *prometheus.HistogramVec, val float64) {
	if m.config.FakeMetrics != nil {
		return
	}
	if hist != nil {
		hist.WithLabelValues(m.config.DisplayModelName).Observe(val)
	}
}

func (m *VLLMMetricsAdapter) reportRunningRequests() {
	if m.runningRequests != nil {
		m.runningRequests.WithLabelValues(m.config.DisplayModelName).Set(float64(m.nRunningReqs))
	}
}

func (m *VLLMMetricsAdapter) reportWaitingRequests() {
	if m.waitingRequests != nil {
		m.waitingRequests.WithLabelValues(m.config.DisplayModelName).Set(float64(m.nWaitingReqs))
	}
}

func (m *VLLMMetricsAdapter) reportKVCacheUsage(value float64) {
	if m.kvCacheUsagePercentage != nil {
		m.kvCacheUsagePercentage.WithLabelValues(m.config.DisplayModelName).Set(value)
	}
}

func (m *VLLMMetricsAdapter) reportLoras() {
	if m.config.FakeMetrics != nil {
		return
	}
	if m.loraInfo == nil {
		return
	}

	var running []string
	m.runningLoras.Range(func(key any, _ any) bool {
		if lora, ok := key.(string); ok {
			running = append(running, lora)
		}
		return true
	})
	var waiting []string
	m.waitingLoras.Range(func(key any, _ any) bool {
		if lora, ok := key.(string); ok {
			waiting = append(waiting, lora)
		}
		return true
	})

	m.loraInfo.WithLabelValues(
		strconv.Itoa(m.config.MaxLoras),
		strings.Join(running, ","),
		strings.Join(waiting, ","),
	).Set(float64(time.Now().Unix()))
}

func (m *VLLMMetricsAdapter) reportPrefixCacheStats(ev PrefixCacheQueried) {
	if m.config.FakeMetrics != nil {
		return
	}
	if m.prefixCacheQueriesTotal != nil {
		m.prefixCacheQueriesTotal.WithLabelValues(m.config.DisplayModelName).Add(float64(ev.QueriedTokens))
	}
	if m.prefixCacheHitsTotal != nil {
		m.prefixCacheHitsTotal.WithLabelValues(m.config.DisplayModelName).Add(float64(ev.CachedPromptTokens))
	}
}

func (m *VLLMMetricsAdapter) recordRequestMetricsOnSuccess(ev RequestSucceeded) {
	m.requestPromptTokens.WithLabelValues(m.config.DisplayModelName).Observe(float64(ev.PromptTokens))
	m.requestGenerationTokens.WithLabelValues(m.config.DisplayModelName).Observe(float64(ev.GenerationTokens))
	m.promptTokensTotal.WithLabelValues(m.config.DisplayModelName).Add(float64(ev.PromptTokens))
	m.generationTokensTotal.WithLabelValues(m.config.DisplayModelName).Add(float64(ev.GenerationTokens))
	if ev.MaxTokens != nil {
		m.requestParamsMaxTokens.WithLabelValues(m.config.DisplayModelName).Observe(float64(*ev.MaxTokens))
	}
	m.requestSuccessTotal.WithLabelValues(m.config.DisplayModelName, ev.FinishReason).Inc()
	if maxGenTokens, err := common.MaxIntSlice(ev.GenTokensPerChoice); err == nil {
		m.maxNumGenerationTokens.WithLabelValues(m.config.DisplayModelName).Observe(float64(maxGenTokens))
	}
}

func (m *VLLMMetricsAdapter) incrementLoraRefCount(lora string, theMap *sync.Map) {
	count := 0
	if value, ok := theMap.Load(lora); ok {
		count = value.(int)
	}
	theMap.Store(lora, count+1)
}

func (m *VLLMMetricsAdapter) decrementLoraRefCount(lora string, theMap *sync.Map) {
	if value, ok := theMap.Load(lora); ok {
		count := value.(int)
		if count > 1 {
			theMap.Store(lora, count-1)
		} else {
			theMap.Delete(lora)
		}
	}
}

// -- Prometheus wiring ------------------------------------------------------

// buildMetrics constructs and registers all Prometheus collectors. It is called once during adapter construction.
func (m *VLLMMetricsAdapter) buildMetrics() error {
	if err := m.createAndRegisterRunningRequestsGauge(); err != nil {
		return err
	}
	if err := m.createAndRegisterWaitingRequestsGauge(); err != nil {
		return err
	}
	if err := m.createAndRegisterKVCacheUsageGauge(); err != nil {
		return err
	}
	if err := m.createAndRegisterLoraInfoGauge(); err != nil {
		return err
	}
	if err := m.createAndRegisterCacheConfigGauge(); err != nil {
		return err
	}
	if err := m.createAndRegisterTTFTHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterTPOTHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterInterTokenLatencyHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterReqTpotHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterE2EReqLatencyHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterReqQueueTimeHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterReqInferenceTimeHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterReqPrefillTimeHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterReqDecodeTimeHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterReqPromptTokensHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterReqGenerationTokensHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterMaxNumGenerationTokensHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterReqParamsMaxTokensHistogram(); err != nil {
		return err
	}
	if err := m.createAndRegisterPromptTokensTotalCounter(); err != nil {
		return err
	}
	if err := m.createAndRegisterGenerationTokensTotalCounter(); err != nil {
		return err
	}
	if err := m.createAndRegisterRequestSuccessTotalCounter(); err != nil {
		return err
	}
	if err := m.createAndRegisterPrefixCacheHitsTotalCounter(); err != nil {
		return err
	}
	if err := m.createAndRegisterPrefixCacheQueriesTotalCounter(); err != nil {
		return err
	}
	return nil
}

func (m *VLLMMetricsAdapter) createAndRegisterRunningRequestsGauge() error {
	m.runningRequests = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: VLLMReqRunningMetricName,
		Help: "Number of requests currently running on GPU.",
	}, modelLabel)
	if err := m.registry.Register(m.runningRequests); err != nil {
		m.logger.Error(err, "prometheus number of running requests gauge register failed")
		return err
	}
	return nil
}

func (m *VLLMMetricsAdapter) createAndRegisterWaitingRequestsGauge() error {
	m.waitingRequests = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: VLLMReqWaitingMetricName,
		Help: "Prometheus metric for the number of queued requests.",
	}, modelLabel)
	if err := m.registry.Register(m.waitingRequests); err != nil {
		m.logger.Error(err, "prometheus number of requests in queue gauge register failed")
		return err
	}
	return nil
}

func (m *VLLMMetricsAdapter) createAndRegisterKVCacheUsageGauge() error {
	m.kvCacheUsagePercentage = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: VLLMKVCacheUsageMetricName,
		Help: "Prometheus metric for the fraction of KV-cache blocks currently in use (from 0 to 1).",
	}, modelLabel)
	if err := m.registry.Register(m.kvCacheUsagePercentage); err != nil {
		m.logger.Error(err, "prometheus kv cache usage percentage gauge register failed")
		return err
	}
	return nil
}

func (m *VLLMMetricsAdapter) createAndRegisterLoraInfoGauge() error {
	m.loraInfo = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: VLLMLoRARequestsMetricName,
		Help: "Running stats on lora requests.",
	}, []string{api.PromLabelMaxLora, api.PromLabelRunningLoraAdapters, api.PromLabelWaitingLoraAdapters})
	if err := m.registry.Register(m.loraInfo); err != nil {
		m.logger.Error(err, "prometheus lora info gauge register failed")
		return err
	}
	return nil
}

func (m *VLLMMetricsAdapter) createAndRegisterCacheConfigGauge() error {
	m.cacheConfig = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: VLLMCacheConfigName,
		Help: "Information of the LLMEngine CacheConfig.",
	}, []string{api.PromLabelCacheBlockSize, api.PromLabelCacheNumGPUBlocks})
	if err := m.registry.Register(m.cacheConfig); err != nil {
		m.logger.Error(err, "prometheus cache config register failed")
		return err
	}
	return nil
}

func (m *VLLMMetricsAdapter) createAndRegisterTTFTHistogram() error {
	m.ttft = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMTTFTMetricName,
		Help:    "Histogram of time to first token in seconds.",
		Buckets: common.TTFTBucketsBoundaries,
	}, modelLabel)
	if err := m.registry.Register(m.ttft); err != nil {
		m.logger.Error(err, "prometheus time to first token histogram register failed")
		return err
	}
	return nil
}

func (m *VLLMMetricsAdapter) createAndRegisterTPOTHistogram() error {
	m.tpot = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMTPOTMetricName,
		Help:    "Histogram of time per output token in seconds.",
		Buckets: common.TPOTBucketsBoundaries,
	}, modelLabel)
	if err := m.registry.Register(m.tpot); err != nil {
		m.logger.Error(err, "prometheus time per output token histogram register failed")
		return err
	}
	return nil
}

func (m *VLLMMetricsAdapter) createAndRegisterInterTokenLatencyHistogram() error {
	m.interTokenLatency = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMInterTokenLatencyMetricName,
		Help:    "Histogram of inter-token latency in seconds.",
		Buckets: common.TPOTBucketsBoundaries,
	}, modelLabel)
	if err := m.registry.Register(m.interTokenLatency); err != nil {
		m.logger.Error(err, "prometheus inter-token latency histogram register failed")
		return err
	}
	return nil
}

func (m *VLLMMetricsAdapter) createAndRegisterReqTpotHistogram() error {
	m.reqTpot = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMReqTPOTMetricName,
		Help:    "Histogram of time_per_output_token_seconds per request.",
		Buckets: common.TPOTBucketsBoundaries,
	}, modelLabel)
	if err := m.registry.Register(m.reqTpot); err != nil {
		m.logger.Error(err, "prometheus time_per_output_token_seconds per request histogram register failed")
		return err
	}
	return nil
}

func (m *VLLMMetricsAdapter) createAndRegisterE2EReqLatencyHistogram() error {
	m.e2eReqLatency = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLME2EReqLatencyMetricName,
		Help:    "Histogram of end to end request latency in seconds.",
		Buckets: common.RequestLatencyBucketsBoundaries,
	}, modelLabel)
	if err := m.registry.Register(m.e2eReqLatency); err != nil {
		m.logger.Error(err, "prometheus e2e request latency histogram register failed")
		return err
	}
	return nil
}

func (m *VLLMMetricsAdapter) createAndRegisterReqQueueTimeHistogram() error {
	m.reqQueueTime = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMReqQueueTimeMetricName,
		Help:    "Histogram of time spent in WAITING phase for request.",
		Buckets: common.RequestLatencyBucketsBoundaries,
	}, modelLabel)
	if err := m.registry.Register(m.reqQueueTime); err != nil {
		m.logger.Error(err, "prometheus request queue time histogram register failed")
		return err
	}
	return nil
}

func (m *VLLMMetricsAdapter) createAndRegisterReqInferenceTimeHistogram() error {
	m.reqInferenceTime = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMReqInferenceTimeMetricName,
		Help:    "Histogram of time spent in RUNNING phase for request.",
		Buckets: common.RequestLatencyBucketsBoundaries,
	}, modelLabel)
	if err := m.registry.Register(m.reqInferenceTime); err != nil {
		m.logger.Error(err, "prometheus request inference time histogram register failed")
		return err
	}
	return nil
}

func (m *VLLMMetricsAdapter) createAndRegisterReqPrefillTimeHistogram() error {
	m.reqPrefillTime = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMPrefillTimeMetricName,
		Help:    "Histogram of time spent in PREFILL phase for request.",
		Buckets: common.RequestLatencyBucketsBoundaries,
	}, modelLabel)
	if err := m.registry.Register(m.reqPrefillTime); err != nil {
		m.logger.Error(err, "prometheus request prefill time histogram register failed")
		return err
	}
	return nil
}

func (m *VLLMMetricsAdapter) createAndRegisterReqDecodeTimeHistogram() error {
	m.reqDecodeTime = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMDecodeTimeMetricName,
		Help:    "Histogram of time spent in DECODE phase for request.",
		Buckets: common.RequestLatencyBucketsBoundaries,
	}, modelLabel)
	if err := m.registry.Register(m.reqDecodeTime); err != nil {
		m.logger.Error(err, "prometheus request decode time histogram register failed")
		return err
	}
	return nil
}

func (m *VLLMMetricsAdapter) createAndRegisterReqPromptTokensHistogram() error {
	m.requestPromptTokens = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMPromptTokensMetricName,
		Help:    "Number of prefill tokens processed.",
		Buckets: Build125Buckets(m.config.MaxModelLen),
	}, modelLabel)
	if err := m.registry.Register(m.requestPromptTokens); err != nil {
		m.logger.Error(err, "prometheus request_prompt_tokens histogram register failed")
		return err
	}
	return nil
}

func (m *VLLMMetricsAdapter) createAndRegisterReqGenerationTokensHistogram() error {
	m.requestGenerationTokens = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMGenerationTokensMetricName,
		Help:    "Number of generation tokens processed.",
		Buckets: Build125Buckets(m.config.MaxModelLen),
	}, modelLabel)
	if err := m.registry.Register(m.requestGenerationTokens); err != nil {
		m.logger.Error(err, "prometheus request_generation_tokens histogram register failed")
		return err
	}
	return nil
}

func (m *VLLMMetricsAdapter) createAndRegisterMaxNumGenerationTokensHistogram() error {
	m.maxNumGenerationTokens = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMMaxNumGenerationTokensMetricName,
		Help:    "Histogram of maximum number of requested generation tokens.",
		Buckets: Build125Buckets(m.config.MaxModelLen),
	}, modelLabel)
	if err := m.registry.Register(m.maxNumGenerationTokens); err != nil {
		m.logger.Error(err, "prometheus max_num_generation_tokens histogram register failed")
		return err
	}
	return nil
}

func (m *VLLMMetricsAdapter) createAndRegisterReqParamsMaxTokensHistogram() error {
	m.requestParamsMaxTokens = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    VLLMParamMaxTokensMetricName,
		Help:    "Histogram of the max_tokens request parameter.",
		Buckets: Build125Buckets(m.config.MaxModelLen),
	}, modelLabel)
	if err := m.registry.Register(m.requestParamsMaxTokens); err != nil {
		m.logger.Error(err, "prometheus request_params_max_tokens histogram register failed")
		return err
	}
	return nil
}

func (m *VLLMMetricsAdapter) createAndRegisterPromptTokensTotalCounter() error {
	m.promptTokensTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: VLLMPromptTokensTotalMetricName,
		Help: "Total number of prompt tokens processed.",
	}, modelLabel)
	if err := m.registry.Register(m.promptTokensTotal); err != nil {
		m.logger.Error(err, "prometheus prompt_tokens_total counter register failed")
		return err
	}
	return nil
}

func (m *VLLMMetricsAdapter) createAndRegisterGenerationTokensTotalCounter() error {
	m.generationTokensTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: VLLMGenerationTokensTotalMetricName,
		Help: "Total number of generated tokens.",
	}, modelLabel)
	if err := m.registry.Register(m.generationTokensTotal); err != nil {
		m.logger.Error(err, "prometheus generation_tokens_total counter register failed")
		return err
	}
	return nil
}

func (m *VLLMMetricsAdapter) createAndRegisterRequestSuccessTotalCounter() error {
	m.requestSuccessTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: VLLMSuccessTotalMetricName,
		Help: "Count of successfully processed requests.",
	}, []string{api.PromLabelModelName, api.PromLabelFinishReason})
	if err := m.registry.Register(m.requestSuccessTotal); err != nil {
		m.logger.Error(err, "prometheus request_success_total counter register failed")
		return err
	}
	return nil
}

func (m *VLLMMetricsAdapter) createAndRegisterPrefixCacheHitsTotalCounter() error {
	m.prefixCacheHitsTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: VLLMPrefixCacheHitsTotalMetricName,
		Help: "Prefix cache hits, in terms of number of cached tokens.",
	}, modelLabel)
	if err := m.registry.Register(m.prefixCacheHitsTotal); err != nil {
		m.logger.Error(err, "prometheus prefix_cache_hits_total counter register failed")
		return err
	}
	return nil
}

func (m *VLLMMetricsAdapter) createAndRegisterPrefixCacheQueriesTotalCounter() error {
	m.prefixCacheQueriesTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: VLLMPrefixCacheQueriesTotalMetricName,
		Help: "Prefix cache queries, in terms of number of queried tokens.",
	}, modelLabel)
	if err := m.registry.Register(m.prefixCacheQueriesTotal); err != nil {
		m.logger.Error(err, "prometheus prefix_cache_queries_total counter register failed")
		return err
	}
	return nil
}

// setInitialValues zeroes gauges that must appear on the first scrape and
// stamps the one-shot cache_config_info and lora_requests_info series so
// output stays consistent with the current metrics.go behavior before any
// request runs. In fake mode the controller's SetInitial (invoked from
// Start) overwrites everything it manages, so the baseline stamped here is
// harmless.
func (m *VLLMMetricsAdapter) setInitialValues() error {
	m.runningRequests.WithLabelValues(m.config.DisplayModelName).Set(0)
	m.waitingRequests.WithLabelValues(m.config.DisplayModelName).Set(0)
	m.kvCacheUsagePercentage.WithLabelValues(m.config.DisplayModelName).Set(0)
	m.cacheConfig.WithLabelValues(
		strconv.Itoa(m.config.TokenBlockSize),
		strconv.Itoa(m.config.KVCacheSize),
	).Set(1)
	m.loraInfo.WithLabelValues(
		strconv.Itoa(m.config.MaxLoras),
		"",
		"",
	).Set(float64(time.Now().Unix()))

	return nil
}

// -- VLLMFakeMetricsApplier implementation ---------------------------------
//
// Each method translates a semantic applier call into a message on the
// per-family channel that owns the matching Prometheus collector. The
// updater goroutines drain those channels and perform the actual
// unregister+recreate+replay. Applier methods are safe to call
// concurrently (channel sends are serialising) and non-blocking under
// normal load: common.WriteToChannel drops with a warning if the buffer
// fills.

func (m *VLLMMetricsAdapter) SetRunningRequests(value float64) {
	common.WriteToChannel(m.runReqChan,
		common.MetricInfo{Value: value, IsFake: true}, m.logger)
}

func (m *VLLMMetricsAdapter) SetWaitingRequests(value float64) {
	common.WriteToChannel(m.waitingReqChan,
		common.MetricInfo{Value: value, IsFake: true}, m.logger)
}

func (m *VLLMMetricsAdapter) SetKVCacheUsagePerc(value float64) {
	common.WriteToChannel(m.kvCacheUsageChan,
		common.MetricInfo{Value: value, IsFake: true}, m.logger)
}

func (m *VLLMMetricsAdapter) SetHistogram(kind VLLMHistogramKind, bucketsBoundaries []float64, samplesCount []int) {
	reset := &HistogramReset{Buckets: bucketsBoundaries, Samples: samplesCount}
	upd := HistogramUpdate{Reset: reset}
	switch kind {
	case VLLMHistTTFT:
		common.WriteToChannel(m.ttftChan, upd, m.logger)
	case VLLMHistTPOT:
		common.WriteToChannel(m.tpotChan, upd, m.logger)
	case VLLMHistInterTokenLatency:
		common.WriteToChannel(m.interTokenLatencyChan, upd, m.logger)
	case VLLMHistE2ERequestLatency:
		common.WriteToChannel(m.e2eReqLatencyChan, upd, m.logger)
	case VLLMHistReqQueueTime:
		common.WriteToChannel(m.reqQueueTimeChan, upd, m.logger)
	case VLLMHistReqInferenceTime:
		common.WriteToChannel(m.reqInferenceTimeChan, upd, m.logger)
	case VLLMHistReqPrefillTime:
		common.WriteToChannel(m.reqPrefillTimeChan, upd, m.logger)
	case VLLMHistReqDecodeTime:
		common.WriteToChannel(m.reqDecodeTimeChan, upd, m.logger)
	case VLLMHistReqTPOT:
		common.WriteToChannel(m.reqTpotChan, upd, m.logger)
	case VLLMHistRequestParamsMaxTokens:
		common.WriteToChannel(m.requestSuccessChan,
			RequestSuccessUpdate{ParamsMaxTokensReset: reset}, m.logger)
	case VLLMHistRequestMaxGenerationTokens:
		common.WriteToChannel(m.requestSuccessChan,
			RequestSuccessUpdate{MaxNumGenTokensReset: reset}, m.logger)
	}
}

func (m *VLLMMetricsAdapter) SetCounter(kind VLLMCounterKind, value int64) {
	upd := PrefixCacheStatUpdate{}
	switch kind {
	case VLLMCounterPrefixCacheQueries:
		upd.ResetQueries = &value
	case VLLMCounterPrefixCacheHits:
		upd.ResetHits = &value
	default:
		return
	}
	common.WriteToChannel(m.prefixCacheStatsChan, upd, m.logger)
}

func (m *VLLMMetricsAdapter) SetSuccessTotalByReason(reasonCounts map[string]int64) {
	common.WriteToChannel(m.requestSuccessChan,
		RequestSuccessUpdate{SuccessTotalReset: &SuccessTotalReset{Reasons: reasonCounts}}, m.logger)
}

func (m *VLLMMetricsAdapter) SetTokenMetric(kind VLLMTokenMetricKind, bucketsBoundaries []float64, samplesCount []int, explicitTotal *int64) {
	common.WriteToChannel(m.requestSuccessChan, RequestSuccessUpdate{
		TokenMetricReset: &TokenMetricReset{
			Kind:          kind,
			Buckets:       bucketsBoundaries,
			Samples:       samplesCount,
			ExplicitTotal: explicitTotal,
		},
	}, m.logger)
}

func (m *VLLMMetricsAdapter) SetLoRAs(maxLoRAs int, entries []common.LorasMetrics) {
	common.WriteToChannel(m.lorasChan,
		LoRAUpdate{Reset: &LoRAReset{MaxLoRAs: maxLoRAs, Entries: entries}}, m.logger)
}

