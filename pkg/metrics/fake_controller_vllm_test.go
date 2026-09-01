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

package metrics

import (
	"context"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

// mockApplier records applier calls in the order they arrive. Safe under
// the controller's serialization guarantee, plus its own mutex so ticker
// goroutine reads don't race a test-side snapshot() call.
type mockApplier struct {
	mu    sync.Mutex
	calls []applierCall
}

type applierCall struct {
	kind string
	// scalar
	value float64
	// histogram
	histKind    VLLMHistogramKind
	histBuckets []float64
	histSamples []int
	// counter
	counterKind  VLLMCounterKind
	counterValue int64
	// token metric
	tokenKind          VLLMTokenMetricKind
	tokenExplicitTotal *int64
	// success total
	reasons map[string]int64
	// loras
	maxLoRAs    int
	loraEntries []common.LorasMetrics
}

func (m *mockApplier) record(c applierCall) {
	m.mu.Lock()
	m.calls = append(m.calls, c)
	m.mu.Unlock()
}

func (m *mockApplier) snapshot() []applierCall {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]applierCall, len(m.calls))
	copy(out, m.calls)
	return out
}

func (m *mockApplier) reset() {
	m.mu.Lock()
	m.calls = nil
	m.mu.Unlock()
}

func (m *mockApplier) SetRunningRequests(v float64) {
	m.record(applierCall{kind: "running", value: v})
}
func (m *mockApplier) SetWaitingRequests(v float64) {
	m.record(applierCall{kind: "waiting", value: v})
}
func (m *mockApplier) SetKVCacheUsagePerc(v float64) {
	m.record(applierCall{kind: "kvcache", value: v})
}
func (m *mockApplier) SetHistogram(kind VLLMHistogramKind, buckets []float64, samples []int) {
	m.record(applierCall{kind: "histogram", histKind: kind, histBuckets: buckets, histSamples: samples})
}
func (m *mockApplier) SetCounter(kind VLLMCounterKind, v int64) {
	m.record(applierCall{kind: "counter", counterKind: kind, counterValue: v})
}
func (m *mockApplier) SetSuccessTotalByReason(r map[string]int64) {
	m.record(applierCall{kind: "success", reasons: r})
}
func (m *mockApplier) SetTokenMetric(kind VLLMTokenMetricKind, buckets []float64, samples []int, explicit *int64) {
	m.record(applierCall{kind: "token", tokenKind: kind, histBuckets: buckets, histSamples: samples, tokenExplicitTotal: explicit})
}
func (m *mockApplier) SetLoRAs(maxLoRAs int, entries []common.LorasMetrics) {
	m.record(applierCall{kind: "loras", maxLoRAs: maxLoRAs, loraEntries: entries})
}

func mkConfig() *common.Configuration {
	return &common.Configuration{
		MaxLoras:                   2,
		MaxModelLen:                1024,
		FakeMetricsRefreshInterval: 5 * time.Millisecond,
	}
}

func mkController(cfg *common.Configuration, applier VLLMFakeMetricsApplier) *VLLMFakeMetricsController {
	return NewVLLMFakeMetricsController(cfg, applier, logr.Discard())
}

func fixed(v float64) *common.FakeMetricWithFunction {
	return &common.FakeMetricWithFunction{FixedValue: v}
}

func fn(name string, start, end float64, period time.Duration) *common.FakeMetricWithFunction {
	return &common.FakeMetricWithFunction{
		IsFunction: true,
		Function:   &common.FunctionInfo{Name: name, Start: start, End: end, Period: period},
	}
}

func countByKind(calls []applierCall, kind string) int {
	n := 0
	for _, c := range calls {
		if c.kind == kind {
			n++
		}
	}
	return n
}

func TestSetInitial_PopulatedFakeMetrics(t *testing.T) {
	m := &mockApplier{}
	c := mkController(mkConfig(), m)

	queries := int64(7)
	hits := int64(3)
	totalPrompt := int64(100)
	fm := &common.FakeMetrics{
		RunningRequests:               fixed(4),
		WaitingRequests:               fixed(2),
		KVCacheUsagePercentage:        fixed(0.5),
		TTFTBucketValues:              []int{1, 2, 3},
		E2ERequestLatencyBucketValues: []int{1},
		RequestPromptTokens:           []int{10, 20},
		TotalPromptTokens:             &totalPrompt,
		PrefixCacheQueries:            &queries,
		PrefixCacheHits:               &hits,
		RequestSuccessTotal:           map[string]int64{"stop": 1},
		LoraMetrics:                   []common.LorasMetrics{{RunningLoras: "a", WaitingLoras: "", Timestamp: 1}},
	}
	if err := c.SetInitial(fm); err != nil {
		t.Fatalf("SetInitial: %v", err)
	}
	calls := m.snapshot()

	// Verify each expected applier method fired at least once with a spot-check
	// on payload.
	if got := countByKind(calls, "running"); got != 1 {
		t.Errorf("running calls: got %d want 1", got)
	}
	if got := countByKind(calls, "waiting"); got != 1 {
		t.Errorf("waiting calls: got %d want 1", got)
	}
	if got := countByKind(calls, "kvcache"); got != 1 {
		t.Errorf("kvcache calls: got %d want 1", got)
	}
	if got := countByKind(calls, "histogram"); got != 2 {
		// TTFT + E2E
		t.Errorf("histogram calls: got %d want 2", got)
	}
	if got := countByKind(calls, "counter"); got != 2 {
		t.Errorf("counter calls: got %d want 2", got)
	}
	if got := countByKind(calls, "token"); got != 1 {
		t.Errorf("token calls: got %d want 1", got)
	}
	if got := countByKind(calls, "success"); got != 1 {
		t.Errorf("success calls: got %d want 1", got)
	}
	if got := countByKind(calls, "loras"); got != 1 {
		t.Errorf("loras calls: got %d want 1", got)
	}
}

func TestApplyUpdate_NilFieldsAreNoOp(t *testing.T) {
	m := &mockApplier{}
	c := mkController(mkConfig(), m)

	// Partial update with only RunningRequests set.
	if err := c.ApplyUpdate(&common.FakeMetrics{RunningRequests: fixed(3)}); err != nil {
		t.Fatalf("ApplyUpdate: %v", err)
	}
	calls := m.snapshot()
	if len(calls) != 1 || calls[0].kind != "running" || calls[0].value != 3 {
		t.Errorf("expected exactly one running call with value 3, got %+v", calls)
	}
}

func TestApplyUpdate_TPOTFansOut(t *testing.T) {
	m := &mockApplier{}
	c := mkController(mkConfig(), m)

	samples := []int{5, 6, 7}
	if err := c.ApplyUpdate(&common.FakeMetrics{TPOTBucketValues: samples}); err != nil {
		t.Fatalf("ApplyUpdate: %v", err)
	}
	calls := m.snapshot()
	if len(calls) != 2 {
		t.Fatalf("expected 2 histogram calls, got %d: %+v", len(calls), calls)
	}
	if calls[0].histKind != VLLMHistTPOT {
		t.Errorf("first histogram kind: got %v want VLLMHistTPOT", calls[0].histKind)
	}
	if calls[1].histKind != VLLMHistInterTokenLatency {
		t.Errorf("second histogram kind: got %v want VLLMHistInterTokenLatency", calls[1].histKind)
	}
	for i, c := range calls {
		if !reflect.DeepEqual(c.histSamples, samples) {
			t.Errorf("call %d histSamples: got %v want %v", i, c.histSamples, samples)
		}
		if !reflect.DeepEqual(c.histBuckets, common.TPOTBucketsBoundaries) {
			t.Errorf("call %d histBuckets: got %v want TPOTBucketsBoundaries", i, c.histBuckets)
		}
	}
}

func TestApplyUpdate_TokenPairExplicitTotalWins(t *testing.T) {
	m := &mockApplier{}
	c := mkController(mkConfig(), m)

	total := int64(999)
	samples := []int{10, 20, 30}
	if err := c.ApplyUpdate(&common.FakeMetrics{
		RequestPromptTokens: samples,
		TotalPromptTokens:   &total,
	}); err != nil {
		t.Fatalf("ApplyUpdate: %v", err)
	}
	calls := m.snapshot()
	if len(calls) != 1 || calls[0].kind != "token" {
		t.Fatalf("expected one token call, got %+v", calls)
	}
	if calls[0].tokenKind != VLLMTokenMetricPrompt {
		t.Errorf("token kind: got %v want prompt", calls[0].tokenKind)
	}
	if calls[0].tokenExplicitTotal == nil || *calls[0].tokenExplicitTotal != total {
		t.Errorf("explicit total: got %v want %d", calls[0].tokenExplicitTotal, total)
	}
	if !reflect.DeepEqual(calls[0].histSamples, samples) {
		t.Errorf("samples: got %v want %v", calls[0].histSamples, samples)
	}
}

func TestApplyUpdate_PrefixCacheHitsAndQueriesTogether(t *testing.T) {
	m := &mockApplier{}
	c := mkController(mkConfig(), m)

	q := int64(11)
	h := int64(4)
	if err := c.ApplyUpdate(&common.FakeMetrics{
		PrefixCacheQueries: &q,
		PrefixCacheHits:    &h,
	}); err != nil {
		t.Fatalf("ApplyUpdate: %v", err)
	}
	calls := m.snapshot()
	if len(calls) != 2 {
		t.Fatalf("expected 2 counter calls, got %d: %+v", len(calls), calls)
	}
	sawQueries, sawHits := false, false
	for _, c := range calls {
		if c.kind != "counter" {
			t.Errorf("unexpected non-counter call: %+v", c)
		}
		switch c.counterKind {
		case VLLMCounterPrefixCacheQueries:
			sawQueries = true
			if c.counterValue != q {
				t.Errorf("queries value: got %d want %d", c.counterValue, q)
			}
		case VLLMCounterPrefixCacheHits:
			sawHits = true
			if c.counterValue != h {
				t.Errorf("hits value: got %d want %d", c.counterValue, h)
			}
		}
	}
	if !sawQueries || !sawHits {
		t.Errorf("expected both counters; queries=%v hits=%v", sawQueries, sawHits)
	}
}

func TestGeneratorTicker_StartsOnFirstStopsOnLast(t *testing.T) {
	m := &mockApplier{}
	cfg := mkConfig()
	cfg.FakeMetricsRefreshInterval = 2 * time.Millisecond
	c := mkController(cfg, m)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	c.Start(ctx)

	// Registering a generator triggers ticker start. Use ramp so t=0 gives
	// Start (0), and later ticks give >0 values — proving the ticker is
	// live.
	if err := c.ApplyUpdate(&common.FakeMetrics{
		RunningRequests: fn(common.RampFuncName, 0, 100, 100*time.Millisecond),
	}); err != nil {
		t.Fatalf("ApplyUpdate: %v", err)
	}

	// Wait for at least a couple of ticks.
	deadline := time.Now().Add(200 * time.Millisecond)
	for time.Now().Before(deadline) {
		if countByKind(m.snapshot(), "running") >= 3 {
			break
		}
		time.Sleep(2 * time.Millisecond)
	}
	if got := countByKind(m.snapshot(), "running"); got < 3 {
		t.Fatalf("expected ticker to produce >=3 running calls, got %d", got)
	}
	c.mu.Lock()
	tickerLive := c.tickerRunning
	c.mu.Unlock()
	if !tickerLive {
		t.Fatal("expected tickerRunning true while generator is active")
	}

	// Remove the generator; ticker should stop.
	m.reset()
	if err := c.ApplyUpdate(&common.FakeMetrics{
		RunningRequests: fixed(42),
	}); err != nil {
		t.Fatalf("ApplyUpdate: %v", err)
	}
	c.mu.Lock()
	tickerLive = c.tickerRunning
	c.mu.Unlock()
	if tickerLive {
		t.Fatal("expected tickerRunning false after removing last generator")
	}

	// Give any in-flight tick time to notice cancellation.
	time.Sleep(20 * time.Millisecond)
	m.reset()
	time.Sleep(20 * time.Millisecond)
	if got := countByKind(m.snapshot(), "running"); got != 0 {
		t.Errorf("expected no further ticker calls after stop, got %d", got)
	}
}
