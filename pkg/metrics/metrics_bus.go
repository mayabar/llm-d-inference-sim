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

	OnRequestReceived(ev RequestReceived)
	OnRequestQueued(ev RequestQueued)
	OnRequestDequeued(ev RequestDequeued)
	OnRequestRunning(ev RequestRunning)
	OnPrefillStarted(ev PrefillStarted)
	OnPrefillEnded(ev PrefillEnded)
	OnDecodeStarted(ev DecodeStarted)
	OnTokenGenerated(ev TokenGenerated)
	OnDecodeEnded(ev DecodeEnded)
	OnRequestSucceeded(ev RequestSucceeded)
	OnRequestFailed(ev RequestFailed)
	OnRequestRejected(ev RequestRejected)
	OnKVCacheUsageChanged(ev KVCacheUsageChanged)
	OnPrefixCacheQueried(ev PrefixCacheQueried)

	ApplyUpdate(update *common.FakeMetrics) error
}

// MetricsBus carries state-change events from producers to the engine
// metrics adapter. Producers push via common.WriteToChannel; the adapter
// runs one drainer goroutine per channel.
type MetricsBus struct {
	adapter  EngineMetricsAdapter
	logger   logr.Logger
	registry *prometheus.Registry

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
}

func (b *MetricsBus) Start(ctx context.Context) error {
	return b.adapter.Start(ctx)
}

// ApplyFakeMetricsUpdate forwards a partial fake-metrics update to the
// adapter's fake controller when the adapter supports it. Non-vLLM adapters
// or adapters not in fake mode return nil; the caller is expected to have
// already gated on config.FakeMetrics != nil.
func (b *MetricsBus) ApplyFakeMetricsUpdate(update *common.FakeMetrics) error {
	if b == nil || update == nil {
		return nil
	}
	return b.adapter.ApplyUpdate(update)
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
	IsLoRA bool
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
	IsLoRA bool
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
	IsLoRA             bool
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
// no token or success counter increments. Err is logged.
type RequestFailed struct {
	BaseEvent
	IsLoRA        bool
	Err           error
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

const (
	waitingUsageState loraUsageState = iota
	runningUsageState
	doneUsageState
)

type loraUsageState int

type loraUsage struct {
	// the lora adapter name
	name string
	// state of the lora usage - waiting/running/done
	state loraUsageState
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

	maxNumberOfRunningRequests := config.MaxNumSeqs * 2
	maxNumberOfWaitingRequests := config.MaxWaitingQueueLength * 2
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
	return mBus, nil
}
