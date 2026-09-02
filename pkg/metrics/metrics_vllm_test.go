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

// End-to-end coverage for the VLLMFakeMetricsApplier surface: each applier
// method sends onto a family channel, an updater goroutine drains, and the
// Prometheus registry ends up in the expected state. Every case waits for
// the write to land in the registry before asserting.

package metrics

import (
	"context"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

const testModel = "test-model"

func newTestAdapter(t *testing.T) (*VLLMMetricsAdapter, context.CancelFunc) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	// Non-nil FakeMetrics so the existing fake/real gates in the updaters
	// accept our writes on the fake path.
	cfg := common.Configuration{
		Model:                 "test-model",
		ServedModelNames:      []string{testModel},
		DisplayModelName:      testModel,
		MaxNumSeqs:            8,
		MaxWaitingQueueLength: 16,
		MaxModelLen:           1024,
		MaxLoras:              2,
		TokenBlockSize:        16,
		KVCacheSize:           1024,
		FakeMetrics:           &common.FakeMetrics{},
	}
	bus := &MetricsBus{registry: prometheus.NewRegistry(), logger: logr.Discard()}
	m, err := NewVLLMMetricsAdapter(ctx, bus, logr.Discard(), cfg)
	if err != nil {
		cancel()
		t.Fatalf("NewVLLMMetricsAdapter: %v", err)
	}
	return m, cancel
}

// waitFor polls cond until it returns true or the timeout elapses.
func waitFor(t *testing.T, timeout time.Duration, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	if !cond() {
		t.Fatalf("condition not met within %v", timeout)
	}
}

// gaugeValue returns the current value of a single-label GaugeVec on
// testModel, or NaN when the series has not been stamped yet.
func gaugeValue(t *testing.T, g *prometheus.GaugeVec) float64 {
	t.Helper()
	ch := make(chan prometheus.Metric, 1)
	g.WithLabelValues(testModel).Collect(ch)
	close(ch)
	m := <-ch
	var dtoMetric dto.Metric
	if err := m.Write(&dtoMetric); err != nil {
		t.Fatalf("gauge write: %v", err)
	}
	return dtoMetric.GetGauge().GetValue()
}

// counterValue returns the current value of a labelled CounterVec entry, or
// 0 when the labels have not been stamped.
func counterValue(t *testing.T, name string, registry *prometheus.Registry, labels map[string]string) float64 {
	t.Helper()
	mfs, err := registry.Gather()
	if err != nil {
		t.Fatalf("Gather: %v", err)
	}
	for _, mf := range mfs {
		if mf.GetName() != name {
			continue
		}
		for _, m := range mf.GetMetric() {
			if labelsMatch(m.GetLabel(), labels) {
				return m.GetCounter().GetValue()
			}
		}
	}
	return 0
}

func labelsMatch(got []*dto.LabelPair, want map[string]string) bool {
	if len(got) != len(want) {
		return false
	}
	for _, lp := range got {
		if want[lp.GetName()] != lp.GetValue() {
			return false
		}
	}
	return true
}

// histogramFor returns the *dto.Histogram for name/model, or nil when not
// yet stamped.
func histogramFor(t *testing.T, name string, registry *prometheus.Registry) *dto.Histogram {
	t.Helper()
	mfs, err := registry.Gather()
	if err != nil {
		t.Fatalf("Gather: %v", err)
	}
	for _, mf := range mfs {
		if mf.GetName() != name {
			continue
		}
		for _, m := range mf.GetMetric() {
			if labelsMatch(m.GetLabel(), map[string]string{"model_name": testModel}) {
				return m.GetHistogram()
			}
		}
	}
	return nil
}

func TestApplier_SetRunningRequests(t *testing.T) {
	m, cancel := newTestAdapter(t)
	defer cancel()
	m.SetRunningRequests(7)
	waitFor(t, time.Second, func() bool { return gaugeValue(t, m.runningRequests) == 7 })
}

func TestApplier_SetWaitingRequests(t *testing.T) {
	m, cancel := newTestAdapter(t)
	defer cancel()
	m.SetWaitingRequests(3)
	waitFor(t, time.Second, func() bool { return gaugeValue(t, m.waitingRequests) == 3 })
}

func TestApplier_SetKVCacheUsagePerc(t *testing.T) {
	m, cancel := newTestAdapter(t)
	defer cancel()
	m.SetKVCacheUsagePerc(0.75)
	waitFor(t, time.Second, func() bool { return gaugeValue(t, m.kvCacheUsagePercentage) == 0.75 })
}

// SetHistogram: full pass across every histogram kind. Each kind gets a
// unique sample distribution that lands entirely in one bucket, and we
// assert the histogram's total count matches.
func TestApplier_SetHistogram_AllKinds(t *testing.T) {
	m, cancel := newTestAdapter(t)
	defer cancel()
	cases := []struct {
		kind       VLLMHistogramKind
		metricName string
		buckets    []float64
		samples    []int
		wantCount  uint64
	}{
		{VLLMHistTTFT, VLLMTTFTMetricName, common.TTFTBucketsBoundaries, []int{2}, 2},
		{VLLMHistTPOT, VLLMTPOTMetricName, common.TPOTBucketsBoundaries, []int{3}, 3},
		{VLLMHistInterTokenLatency, VLLMInterTokenLatencyMetricName, common.TPOTBucketsBoundaries, []int{4}, 4},
		{VLLMHistE2ERequestLatency, VLLME2EReqLatencyMetricName, common.RequestLatencyBucketsBoundaries, []int{5}, 5},
		{VLLMHistReqQueueTime, VLLMReqQueueTimeMetricName, common.RequestLatencyBucketsBoundaries, []int{6}, 6},
		{VLLMHistReqInferenceTime, VLLMReqInferenceTimeMetricName, common.RequestLatencyBucketsBoundaries, []int{7}, 7},
		{VLLMHistReqPrefillTime, VLLMPrefillTimeMetricName, common.RequestLatencyBucketsBoundaries, []int{8}, 8},
		{VLLMHistReqDecodeTime, VLLMDecodeTimeMetricName, common.RequestLatencyBucketsBoundaries, []int{9}, 9},
		{VLLMHistReqTPOT, VLLMReqTPOTMetricName, common.TPOTBucketsBoundaries, []int{10}, 10},
		{VLLMHistRequestParamsMaxTokens, VLLMParamMaxTokensMetricName, Build125Buckets(1024), []int{11}, 11},
		{VLLMHistRequestMaxGenerationTokens, VLLMMaxNumGenerationTokensMetricName, Build125Buckets(1024), []int{12}, 12},
	}
	for _, tc := range cases {
		m.SetHistogram(tc.kind, tc.buckets, tc.samples)
	}
	for _, tc := range cases {
		want := tc.wantCount
		name := tc.metricName
		waitFor(t, 2*time.Second, func() bool {
			h := histogramFor(t, name, m.bus.registry)
			return h != nil && h.GetSampleCount() == want
		})
	}
}

func TestApplier_SetCounter(t *testing.T) {
	m, cancel := newTestAdapter(t)
	defer cancel()
	m.SetCounter(VLLMCounterPrefixCacheQueries, 42)
	m.SetCounter(VLLMCounterPrefixCacheHits, 17)
	labels := map[string]string{"model_name": testModel}
	waitFor(t, time.Second, func() bool {
		return counterValue(t, VLLMPrefixCacheQueriesTotalMetricName, m.bus.registry, labels) == 42 &&
			counterValue(t, VLLMPrefixCacheHitsTotalMetricName, m.bus.registry, labels) == 17
	})
}

func TestApplier_SetSuccessTotalByReason(t *testing.T) {
	m, cancel := newTestAdapter(t)
	defer cancel()
	m.SetSuccessTotalByReason(map[string]int64{"stop": 5, "length": 2})
	stopLabels := map[string]string{"model_name": testModel, "finish_reason": "stop"}
	lengthLabels := map[string]string{"model_name": testModel, "finish_reason": "length"}
	waitFor(t, time.Second, func() bool {
		return counterValue(t, VLLMSuccessTotalMetricName, m.bus.registry, stopLabels) == 5 &&
			counterValue(t, VLLMSuccessTotalMetricName, m.bus.registry, lengthLabels) == 2
	})
}

// SetTokenMetric with samples but no explicit total: the counter is
// seeded from the derived histogram sum.
func TestApplier_SetTokenMetric_HistogramOnly(t *testing.T) {
	m, cancel := newTestAdapter(t)
	defer cancel()
	buckets := []float64{10, 100}
	samples := []int{1, 2} // total = 1*10 + 2*100 = 210
	m.SetTokenMetric(VLLMTokenMetricPrompt, buckets, samples, nil)
	labels := map[string]string{"model_name": testModel}
	waitFor(t, time.Second, func() bool {
		h := histogramFor(t, VLLMPromptTokensMetricName, m.bus.registry)
		return h != nil && h.GetSampleCount() == 3 &&
			counterValue(t, VLLMPromptTokensTotalMetricName, m.bus.registry, labels) == 210
	})
}

// SetTokenMetric with explicit total overrides the derived sum.
func TestApplier_SetTokenMetric_ExplicitTotalWins(t *testing.T) {
	m, cancel := newTestAdapter(t)
	defer cancel()
	total := int64(9999)
	m.SetTokenMetric(VLLMTokenMetricGeneration, []float64{10, 100}, []int{1, 2}, &total)
	labels := map[string]string{"model_name": testModel}
	waitFor(t, time.Second, func() bool {
		return counterValue(t, VLLMGenerationTokensTotalMetricName, m.bus.registry, labels) == 9999
	})
}

// SetTokenMetric with only ExplicitTotal touches the counter but leaves
// the histogram untouched (sample count stays zero).
func TestApplier_SetTokenMetric_ExplicitTotalOnly(t *testing.T) {
	m, cancel := newTestAdapter(t)
	defer cancel()
	total := int64(500)
	m.SetTokenMetric(VLLMTokenMetricPrompt, nil, nil, &total)
	labels := map[string]string{"model_name": testModel}
	waitFor(t, time.Second, func() bool {
		return counterValue(t, VLLMPromptTokensTotalMetricName, m.bus.registry, labels) == 500
	})
	if h := histogramFor(t, VLLMPromptTokensMetricName, m.bus.registry); h != nil && h.GetSampleCount() != 0 {
		t.Errorf("histogram should be untouched, got sample count %d", h.GetSampleCount())
	}
}

func TestApplier_SetLoRAs_Entries(t *testing.T) {
	m, cancel := newTestAdapter(t)
	defer cancel()
	entries := []common.LorasMetrics{
		{RunningLoras: "a,b", WaitingLoras: "c", Timestamp: 111},
		{RunningLoras: "d", WaitingLoras: "", Timestamp: 222},
	}
	m.SetLoRAs(4, entries)
	waitFor(t, time.Second, func() bool {
		mfs, err := m.bus.registry.Gather()
		if err != nil {
			return false
		}
		for _, mf := range mfs {
			if mf.GetName() != VLLMLoRARequestsMetricName {
				continue
			}
			return len(mf.GetMetric()) == 2
		}
		return false
	})
}

// Empty entries records a single zero-adapter row.
func TestApplier_SetLoRAs_Empty(t *testing.T) {
	m, cancel := newTestAdapter(t)
	defer cancel()
	m.SetLoRAs(2, nil)
	waitFor(t, time.Second, func() bool {
		mfs, err := m.bus.registry.Gather()
		if err != nil {
			return false
		}
		for _, mf := range mfs {
			if mf.GetName() != VLLMLoRARequestsMetricName {
				continue
			}
			metrics := mf.GetMetric()
			if len(metrics) != 1 {
				return false
			}
			// Expect a stamped-recently timestamp (>0) with empty running/waiting.
			return metrics[0].GetGauge().GetValue() > 0
		}
		return false
	})
}
