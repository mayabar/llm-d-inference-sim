/*
Copyright 2026 The llm-d-inference-sim Authors.

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

// Semantic surface for vLLM fake-metrics application. VLLMFakeMetricsApplier
// is the sink that VLLMFakeMetricsController drives: each method describes a
// mutation on a vLLM metric family, leaving the backend free to decide how
// to reach that state (Prometheus collector reset, in-memory value swap, a
// test double recording calls).
//
// The surface is vLLM-shaped because common.FakeMetrics is vLLM-shaped. A
// future engine that supports fake metrics will land its own applier and
// controller alongside this pair.

package metrics

import (
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

// VLLMHistogramKind identifies a vLLM fake-metrics histogram family.
type VLLMHistogramKind int

const (
	VLLMHistTTFT VLLMHistogramKind = iota
	VLLMHistTPOT
	VLLMHistInterTokenLatency
	VLLMHistE2ERequestLatency
	VLLMHistReqQueueTime
	VLLMHistReqInferenceTime
	VLLMHistReqPrefillTime
	VLLMHistReqDecodeTime
	VLLMHistReqTPOT
	VLLMHistRequestParamsMaxTokens
	VLLMHistRequestMaxGenerationTokens
)

// VLLMCounterKind identifies a single-valued fake-metrics counter family.
// Counters keyed by an extra label (request_success_total by reason) get
// their own applier method.
type VLLMCounterKind int

const (
	VLLMCounterPrefixCacheQueries VLLMCounterKind = iota
	VLLMCounterPrefixCacheHits
)

// VLLMTokenMetricKind identifies one leg of the histogram+total counter
// pairs the fake config exposes for prompt / generation token counts.
type VLLMTokenMetricKind int

const (
	VLLMTokenMetricPrompt VLLMTokenMetricKind = iota
	VLLMTokenMetricGeneration
)

// VLLMFakeMetricsApplier is the sink the vLLM fake-metrics controller drives.
// Each method is a semantic mutation; the implementation decides how to
// realise it against its backend.
//
// SetHistogram, SetCounter, SetTokenMetric, SetSuccessTotalByReason and
// SetLoRAs are reset-shaped: the caller states the new absolute state for
// the family and the backend must discard accumulated observations before
// recording the supplied values.
//
// Concurrency: callers must serialize invocations. The controller that
// drives this interface holds a mutex around its ticker tick, SetInitial,
// and ApplyUpdate paths, so at most one applier method is in flight at any
// time and implementations need not be internally thread-safe.
// Implementations that happen to be safe under concurrent use (e.g. a
// channel-send adapter) are welcome to relax this, but the interface makes
// no such guarantee.
type VLLMFakeMetricsApplier interface {
	SetRunningRequests(value float64)
	SetWaitingRequests(value float64)
	SetKVCacheUsagePerc(value float64)

	// SetHistogram replaces the histogram identified by kind with a fresh
	// collector whose observations correspond to samplesCount over the
	// provided bucket boundaries (see initFakeHistogram semantics).
	SetHistogram(kind VLLMHistogramKind, bucketsBoundaries []float64, samplesCount []int)

	// SetCounter replaces the counter identified by kind and adds value.
	SetCounter(kind VLLMCounterKind, value int64)

	// SetSuccessTotalByReason replaces the request_success_total counter and
	// adds one entry per (reason, count) pair.
	SetSuccessTotalByReason(reasonCounts map[string]int64)

	// SetTokenMetric replaces the histogram+total counter pair for kind.
	// If samplesCount is nil the histogram is left untouched. If
	// explicitTotal is non-nil it overrides the histogram-derived sum;
	// otherwise the derived sum is used. When both are nil the counter is
	// left untouched too.
	SetTokenMetric(kind VLLMTokenMetricKind, bucketsBoundaries []float64, samplesCount []int, explicitTotal *int64)

	// SetLoRAs replaces the lora_requests_info gauge with the supplied
	// entries. An empty entries slice records a single zero-adapter row
	// stamped with the current time (matching the fake-metrics default).
	SetLoRAs(maxLoRAs int, entries []common.LorasMetrics)
}
