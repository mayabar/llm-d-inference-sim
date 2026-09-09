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

package metrics

import (
	"context"
	"sync"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/prometheus/client_golang/prometheus"
)

// EngineMetricsAdapter consumes state-change events from a MetricsBus and
// turns them into engine-specific metric observations. Start spawns one
// goroutine per bus channel; each drainer dispatches events to the matching
// On<Event> method. Handlers run serially per channel, so a slow handler
// for one event kind cannot block another. Registry returns the Prometheus
// registry exposed on /metrics; adapters that do not publish Prometheus
// return nil.
type EngineMetricsAdapter interface {
	Start(ctx context.Context) error
	Close() error

	onRequestReceived(ev RequestReceived)
	onRequestQueued(ev RequestQueued)
	onRequestDequeued(ev RequestDequeued)
	onRequestRunning(ev RequestRunning)
	onPrefillStarted(ev PrefillStarted)
	onPrefillEnded(ev PrefillEnded)
	onDecodeStarted(ev DecodeStarted)
	onTokenGenerated(ev TokenGenerated)
	onDecodeEnded(ev DecodeEnded)
	onRequestSucceeded(ev RequestSucceeded)
	onRequestFailed(ev RequestFailed)
	onRequestRejected(ev RequestRejected)
	onKVCacheUsageChanged(ev KVCacheUsageChanged)
	onPrefixCacheQueried(ev PrefixCacheQueried)
	onLoRASetsChanged(ev loraSetsChanged)

	applyUpdate(update *common.FakeMetrics) error
}

// MetricsBus carries state-change events from producers to the engine
// metrics adapter. Producers push via common.WriteToChannel; the adapter
// runs one drainer goroutine per channel.
type MetricsBus struct {
	adapter  EngineMetricsAdapter
	logger   logr.Logger
	registry *prometheus.Registry

	// Per-LoRA waiting/running request counts, mutated only by the LoRAChanged subscriber
	runningLoras sync.Map
	waitingLoras sync.Map

	RequestQueued    common.Channel[RequestQueued]
	RequestDequeued  common.Channel[RequestDequeued]
	RequestRunning   common.Channel[RequestRunning]
	PrefillStarted   common.Channel[PrefillStarted]
	PrefillEnded     common.Channel[PrefillEnded]
	DecodeStarted    common.Channel[DecodeStarted]
	TokenGenerated   common.Channel[TokenGenerated]
	DecodeEnded      common.Channel[DecodeEnded]
	RequestSucceeded common.Channel[RequestSucceeded]
	RequestFailed    common.Channel[RequestFailed]
	KVCacheUsage     common.Channel[KVCacheUsageChanged]
	PrefixCacheQuery common.Channel[PrefixCacheQueried]
	// LoRAChanged carries all LoRA request transitions on one channel to preserve per-request ordering.
	LoRAChanged common.Channel[LoRAChanged]
	// loraSetsChanged is emitted by the bus after each LoRAChanged event with the per-LoRA waiting/running counts.
	loraSetsChanged common.Channel[loraSetsChanged]
}

func (b *MetricsBus) Start(ctx context.Context) error {
	// start LoRA counter loop before starting the adapter,
	// so that the first LoRAChanged event is processed before the first LoRASetsChanged event.
	go b.loraCounterLoop(ctx)

	return b.adapter.Start(ctx)
}

// loraCounterLoop subscribes to LoRAChanged, mutates the per-LoRA waiting/running
// counters, and forwards a snapshot on LoRASetsChanged.
func (b *MetricsBus) loraCounterLoop(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case ev := <-b.LoRAChanged.Channel:
			switch ev.State {
			case LoRAWaiting:
				b.incrementLoraRefCount(ev.Model, &b.waitingLoras)
			case LoRARunning:
				b.decrementLoraRefCount(ev.Model, &b.waitingLoras)
				b.incrementLoraRefCount(ev.Model, &b.runningLoras)
			case LoRADone:
				b.decrementLoraRefCount(ev.Model, &b.runningLoras)
			default:
				// invalid event
				continue
			}

			common.WriteToChannel(b.loraSetsChanged, b.snapshotLoraSets(ev.IsFake), b.logger)
		}
	}
}

func (b *MetricsBus) snapshotLoraSets(isFake bool) loraSetsChanged {
	running := make(map[string]int)
	b.runningLoras.Range(func(k, v any) bool {
		running[k.(string)] = v.(int)
		return true
	})
	waiting := make(map[string]int)
	b.waitingLoras.Range(func(k, v any) bool {
		waiting[k.(string)] = v.(int)
		return true
	})
	return loraSetsChanged{
		BaseEvent: BaseEvent{IsFake: isFake},
		Running:   running,
		Waiting:   waiting,
	}
}

func (b *MetricsBus) incrementLoraRefCount(lora string, theMap *sync.Map) {
	count := 0
	if value, ok := theMap.Load(lora); ok {
		count = value.(int)
	}
	theMap.Store(lora, count+1)
}

func (b *MetricsBus) decrementLoraRefCount(lora string, theMap *sync.Map) {
	if value, ok := theMap.Load(lora); ok {
		count := value.(int)
		if count > 1 {
			theMap.Store(lora, count-1)
		} else {
			theMap.Delete(lora)
		}
	}
}

// ApplyFakeMetricsUpdate forwards a partial fake-metrics update to the
// adapter's fake controller when the adapter supports it. Non-vLLM adapters
// or adapters not in fake mode return nil; the caller is expected to have
// already gated on config.FakeMetrics != nil.
func (b *MetricsBus) ApplyFakeMetricsUpdate(update *common.FakeMetrics) error {
	if b == nil || update == nil {
		return nil
	}
	return b.adapter.applyUpdate(update)
}

// -- Events -----------------------------------------------------------------
//
// Every duration field is seconds (float64), pre-computed by the producer
// at the current time.Since(...) call site. Adapters never store timestamps
// and never compute diffs.
type BaseEvent struct {
	IsFake bool
	Model  string
}

// RequestReceived fires when a request enters HandleRequest, before queue admission.
// State marker: not consumed by any exposed metric today.
type RequestReceived struct {
	BaseEvent
}

// RequestRejected fires when a request is refused before it can be queued
// (queue full, invalid model, injected failure).
// State marker: not consumed by any exposed metric today.
type RequestRejected struct {
	BaseEvent
	Err error
}

// RequestQueued fires when a request is admitted to the waiting queue.
// Drives num_requests_waiting (+1) and the LoRA waiting-set add.
type RequestQueued struct {
	BaseEvent
}

// RequestDequeued fires when a request is pulled from the waiting queue.
// Drives num_requests_waiting (-1) and request_queue_time_seconds.
type RequestDequeued struct {
	BaseEvent
	QueueTime float64 // seconds
}

// RequestRunning fires when a worker begins processing a request, before
// prefill. Drives num_requests_running (+1) and the LoRA waiting->running move.
type RequestRunning struct {
	BaseEvent
}

// PrefillStarted fires at the start of simulated prefill. State marker.
type PrefillStarted struct {
	BaseEvent
}

// PrefillEnded fires after the simulated prefill delay. Drives
// request_prefill_time_seconds and time_to_first_token_seconds (same
// value observed on both histograms).
type PrefillEnded struct {
	BaseEvent
	PrefillDuration float64 // seconds
}

// DecodeStarted fires before the per-token generation loop. State marker.
type DecodeStarted struct {
	BaseEvent
}

// TokenGenerated fires once per generated token (from the second token
// onward). InterTokenLatency is the elapsed time since the previous
// token, computed by the producer.
type TokenGenerated struct {
	BaseEvent
	InterTokenLatency float64 // seconds
}

// DecodeEnded fires after the last token. Drives request_decode_time_seconds
// (Observe DecodeDuration) and request_time_per_output_token_seconds
// (Observe DecodeDuration/GenerationTokens when GenerationTokens > 0).
type DecodeEnded struct {
	BaseEvent
	GenerationTokens int
	DecodeDuration   float64 // seconds
}

// RequestSucceeded is the terminal event for a request that produced a
// response. Drives all token, success, and both request-level latency
// histograms, plus num_requests_running (-1) and LoRA running-set removal.
type RequestSucceeded struct {
	BaseEvent
	PromptTokens       int
	GenerationTokens   int
	GenTokensPerChoice []int
	MaxTokens          *int64
	FinishReason       string
	E2ELatency         float64 // seconds
	InferenceTime      float64 // seconds
}

// RequestFailed is the terminal event for a request that errored out. Same
// running-counter, LoRA, and latency bookkeeping as RequestSucceeded, but
// no token or success counter increments.
type RequestFailed struct {
	BaseEvent
	E2ELatency    float64 // seconds
	InferenceTime float64 // seconds
}

// KVCacheUsageChanged fires when block-cache utilization changes.
// Cache-wide (not per-request); Model is empty.
type KVCacheUsageChanged struct {
	BaseEvent
	KVCacheUsagePerc float64
}

// PrefixCacheQueried fires on prefix-cache lookup at request start.
// Cache-wide; Model is empty.
type PrefixCacheQueried struct {
	BaseEvent
	QueriedTokens      int
	CachedPromptTokens int
}

// LoRAState identifies the transition a LoRA-carrying request has made.
type LoRAState int

const (
	// LoRAWaiting: request has just been enqueued for a LoRA.
	LoRAWaiting LoRAState = iota
	// LoRARunning: request for a LoRA has been picked up by a worker.
	LoRARunning
	// LoRADone: request for a LoRA has completed (success or failure).
	LoRADone
)

// LoRAChanged signals a waiting/running/done transition for a LoRA request.
type LoRAChanged struct {
	BaseEvent
	State LoRAState
}

// loraSetsChanged is emitted after the bus applies a LoRAChanged event; it
// carries the current per-LoRA waiting and running request counts. Adapters
// derive labels (e.g. lora_requests_info) from these maps.
type loraSetsChanged struct {
	BaseEvent
	Running map[string]int
	Waiting map[string]int
}

// channelCapacities returns the buffered-channel sizes derived from config,
// shared by the bus and every EngineMetricsAdapter so buffer sizing stays
// consistent across both.
func channelCapacities(config common.Configuration) (running, waiting, requests int) {
	running = config.MaxNumSeqs * 2
	waiting = config.MaxWaitingQueueLength * 2
	requests = (config.MaxNumSeqs + config.MaxWaitingQueueLength) * 2
	return
}

// --------------------------------
func NewMetricsBus(ctx context.Context, config common.Configuration, registry *prometheus.Registry, logger logr.Logger) (*MetricsBus, error) {
	mBus := &MetricsBus{
		registry: registry,
		logger:   logger,
	}

	// TODO create metrics adapter based on config.EngineType (vllm, sglang, etc.)
	adapter, err := NewVLLMMetricsAdapter(ctx, mBus, logger, config)
	if err != nil {
		return nil, err
	}

	mBus.adapter = adapter

	// create channels with capacity based on config
	done := ctx.Done()

	maxNumberOfRunningRequests, maxNumberOfWaitingRequests, _ := channelCapacities(config)
	maxNumberOfTokens := maxNumberOfRunningRequests * config.MaxModelLen

	mBus.RequestQueued = common.Channel[RequestQueued]{
		Channel: make(chan RequestQueued, maxNumberOfWaitingRequests),
		Name:    "bus.RequestQueued",
		Done:    done,
	}
	mBus.RequestDequeued = common.Channel[RequestDequeued]{
		Channel: make(chan RequestDequeued, maxNumberOfWaitingRequests),
		Name:    "bus.RequestDequeued",
		Done:    done,
	}
	mBus.RequestRunning = common.Channel[RequestRunning]{
		Channel: make(chan RequestRunning, maxNumberOfRunningRequests),
		Name:    "bus.RequestRunning",
		Done:    done,
	}
	mBus.PrefillStarted = common.Channel[PrefillStarted]{
		Channel: make(chan PrefillStarted, maxNumberOfRunningRequests),
		Name:    "bus.PrefillStarted",
		Done:    done,
	}
	mBus.PrefillEnded = common.Channel[PrefillEnded]{
		Channel: make(chan PrefillEnded, maxNumberOfRunningRequests),
		Name:    "bus.PrefillEnded",
		Done:    done,
	}
	mBus.DecodeStarted = common.Channel[DecodeStarted]{
		Channel: make(chan DecodeStarted, maxNumberOfRunningRequests),
		Name:    "bus.DecodeStarted",
		Done:    done,
	}
	mBus.TokenGenerated = common.Channel[TokenGenerated]{
		Channel: make(chan TokenGenerated, maxNumberOfTokens),
		Name:    "bus.TokenGenerated",
		Done:    done,
	}
	mBus.DecodeEnded = common.Channel[DecodeEnded]{
		Channel: make(chan DecodeEnded, maxNumberOfRunningRequests),
		Name:    "bus.DecodeEnded",
		Done:    done,
	}
	mBus.RequestSucceeded = common.Channel[RequestSucceeded]{
		Channel: make(chan RequestSucceeded, maxNumberOfRunningRequests),
		Name:    "bus.RequestSucceeded",
		Done:    done,
	}
	mBus.RequestFailed = common.Channel[RequestFailed]{
		Channel: make(chan RequestFailed, maxNumberOfRunningRequests),
		Name:    "bus.RequestFailed",
		Done:    done,
	}
	mBus.KVCacheUsage = common.Channel[KVCacheUsageChanged]{
		Channel: make(chan KVCacheUsageChanged, maxNumberOfRunningRequests),
		Name:    "bus.KVCacheUsage",
		Done:    done,
	}
	mBus.PrefixCacheQuery = common.Channel[PrefixCacheQueried]{
		Channel: make(chan PrefixCacheQueried, maxNumberOfRunningRequests),
		Name:    "bus.PrefixCacheQuery",
		Done:    done,
	}
	mBus.LoRAChanged = common.Channel[LoRAChanged]{
		Channel: make(chan LoRAChanged, maxNumberOfWaitingRequests+maxNumberOfRunningRequests),
		Name:    "bus.LoRAChanged",
		Done:    done,
	}
	mBus.loraSetsChanged = common.Channel[loraSetsChanged]{
		Channel: make(chan loraSetsChanged, maxNumberOfWaitingRequests+maxNumberOfRunningRequests),
		Name:    "bus.LoRASetsChanged",
		Done:    done,
	}
	return mBus, nil
}
