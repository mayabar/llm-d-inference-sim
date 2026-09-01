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

// State machine that drives fake vLLM metrics through a VLLMFakeMetricsApplier.
// The controller reads common.FakeMetrics, decides which applier methods to
// invoke (including the two-histogram TPOT/InterTokenLatency fan-out and the
// token histogram + total counter pair), and runs the periodic ticker for
// function-driven scalar gauges. It holds no Prometheus references.

package metrics

import (
	"context"
	"sync"
	"time"

	"github.com/go-logr/logr"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

// Internal keys for the active-generator map. Kept private so the applier
// interface has no scalar-gauge enum.
const (
	genKeyRunning = "running"
	genKeyWaiting = "waiting"
	genKeyKVCache = "kvcache"
)

type activeGenerator struct {
	fn         Generator
	params     *common.FunctionInfo
	roundToInt bool
	apply      func(v float64)
}

// VLLMFakeMetricsController drives fake-metrics application against a
// VLLMFakeMetricsApplier. All public methods are safe to call concurrently;
// the controller serializes them with a single mutex so at most one applier
// call is in flight at a time.
type VLLMFakeMetricsController struct {
	config  *common.Configuration
	applier VLLMFakeMetricsApplier
	logger  logr.Logger

	mu sync.Mutex

	generators map[string]activeGenerator

	// Ticker lifecycle. tickerRunning is true iff a goroutine spawned by
	// startTickerLocked is live. started is set by Start and gates whether
	// generator-set transitions may spawn a ticker.
	started       bool
	rootCtx       context.Context
	tickerRunning bool
	tickerCancel  context.CancelFunc
	tickerStart   time.Time
}

// NewVLLMFakeMetricsController builds a controller. config is retained by
// reference so field lookups (MaxLoras, MaxModelLen, FakeMetricsRefreshInterval)
// reflect the caller's current configuration snapshot.
func NewVLLMFakeMetricsController(config *common.Configuration, applier VLLMFakeMetricsApplier, logger logr.Logger) *VLLMFakeMetricsController {
	return &VLLMFakeMetricsController{
		config:     config,
		applier:    applier,
		logger:     logger,
		generators: make(map[string]activeGenerator),
	}
}

// SetInitial applies the initial FakeMetrics. LoraMetrics is treated as an
// empty slice when nil so the default zero-adapter row gets registered
// (matching the existing simulator behavior).
func (c *VLLMFakeMetricsController) SetInitial(initial *common.FakeMetrics) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	fm := *initial
	if fm.LoraMetrics == nil {
		fm.LoraMetrics = []common.LorasMetrics{}
	}
	return c.applyLocked(&fm)
}

// ApplyUpdate applies a partial update. Nil fields are skipped; non-nil
// fields are treated as the new absolute state for that family and reach
// the applier as reset-shaped calls.
func (c *VLLMFakeMetricsController) ApplyUpdate(update *common.FakeMetrics) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.applyLocked(update)
}

// Start kicks off the ticker for function-driven scalar gauges if any are
// currently active, and enables further generator-set transitions to
// start/stop it. ctx cancellation stops the ticker.
func (c *VLLMFakeMetricsController) Start(ctx context.Context) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.started = true
	c.rootCtx = ctx
	if len(c.generators) > 0 && !c.tickerRunning {
		c.startTickerLocked()
	}
}

// Close stops the ticker if running. Safe to call multiple times.
func (c *VLLMFakeMetricsController) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.stopTickerLocked()
	c.started = false
}

func (c *VLLMFakeMetricsController) applyLocked(update *common.FakeMetrics) error {
	generatorsWereEmpty := len(c.generators) == 0

	if update.RunningRequests != nil {
		c.applyScalarLocked(genKeyRunning, update.RunningRequests, c.applier.SetRunningRequests, true)
	}
	if update.WaitingRequests != nil {
		c.applyScalarLocked(genKeyWaiting, update.WaitingRequests, c.applier.SetWaitingRequests, true)
	}
	if update.KVCacheUsagePercentage != nil {
		c.applyScalarLocked(genKeyKVCache, update.KVCacheUsagePercentage, c.applier.SetKVCacheUsagePerc, false)
	}

	if update.TTFTBucketValues != nil {
		c.applier.SetHistogram(VLLMHistTTFT, common.TTFTBucketsBoundaries, update.TTFTBucketValues)
	}
	if update.TPOTBucketValues != nil {
		c.applier.SetHistogram(VLLMHistTPOT, common.TPOTBucketsBoundaries, update.TPOTBucketValues)
		c.applier.SetHistogram(VLLMHistInterTokenLatency, common.TPOTBucketsBoundaries, update.TPOTBucketValues)
	}
	if update.E2ERequestLatencyBucketValues != nil {
		c.applier.SetHistogram(VLLMHistE2ERequestLatency, common.RequestLatencyBucketsBoundaries, update.E2ERequestLatencyBucketValues)
	}
	if update.ReqQueueTimeBucketValues != nil {
		c.applier.SetHistogram(VLLMHistReqQueueTime, common.RequestLatencyBucketsBoundaries, update.ReqQueueTimeBucketValues)
	}
	if update.ReqInfTimeBucketValues != nil {
		c.applier.SetHistogram(VLLMHistReqInferenceTime, common.RequestLatencyBucketsBoundaries, update.ReqInfTimeBucketValues)
	}
	if update.ReqPrefillTimeBucketValues != nil {
		c.applier.SetHistogram(VLLMHistReqPrefillTime, common.RequestLatencyBucketsBoundaries, update.ReqPrefillTimeBucketValues)
	}
	if update.ReqDecodeTimeBucketValues != nil {
		c.applier.SetHistogram(VLLMHistReqDecodeTime, common.RequestLatencyBucketsBoundaries, update.ReqDecodeTimeBucketValues)
	}
	if update.ReqTPOTBucketValues != nil {
		c.applier.SetHistogram(VLLMHistReqTPOT, common.TPOTBucketsBoundaries, update.ReqTPOTBucketValues)
	}

	tokenBuckets := Build125Buckets(c.config.MaxModelLen)

	if update.RequestParamsMaxTokens != nil {
		c.applier.SetHistogram(VLLMHistRequestParamsMaxTokens, tokenBuckets, update.RequestParamsMaxTokens)
	}
	if update.RequestMaxGenerationTokens != nil {
		c.applier.SetHistogram(VLLMHistRequestMaxGenerationTokens, tokenBuckets, update.RequestMaxGenerationTokens)
	}

	if update.RequestPromptTokens != nil || update.TotalPromptTokens != nil {
		c.applier.SetTokenMetric(VLLMTokenMetricPrompt, tokenBuckets, update.RequestPromptTokens, update.TotalPromptTokens)
	}
	if update.RequestGenerationTokens != nil || update.TotalGenerationTokens != nil {
		c.applier.SetTokenMetric(VLLMTokenMetricGeneration, tokenBuckets, update.RequestGenerationTokens, update.TotalGenerationTokens)
	}

	if update.PrefixCacheQueries != nil {
		c.applier.SetCounter(VLLMCounterPrefixCacheQueries, *update.PrefixCacheQueries)
	}
	if update.PrefixCacheHits != nil {
		c.applier.SetCounter(VLLMCounterPrefixCacheHits, *update.PrefixCacheHits)
	}

	if update.RequestSuccessTotal != nil {
		c.applier.SetSuccessTotalByReason(update.RequestSuccessTotal)
	}

	if update.LoraMetrics != nil {
		c.applier.SetLoRAs(c.config.MaxLoras, update.LoraMetrics)
	}

	generatorsAreEmpty := len(c.generators) == 0
	if c.started {
		switch {
		case generatorsWereEmpty && !generatorsAreEmpty:
			c.startTickerLocked()
		case !generatorsWereEmpty && generatorsAreEmpty:
			c.stopTickerLocked()
		}
	}

	return nil
}

// applyScalarLocked either registers/replaces the generator for key or
// removes it, and calls apply with the resulting seed value. Matches the
// current setFakeMetricWithFunction semantics: on generator registration
// the value reported is genFun(params, 0); on fixed-value the FixedValue is
// used.
func (c *VLLMFakeMetricsController) applyScalarLocked(key string, fm *common.FakeMetricWithFunction, apply func(float64), roundToInt bool) {
	if fm.IsFunction {
		gen := activeGenerator{
			fn:         Dispatch(fm.Function.Name),
			params:     fm.Function,
			roundToInt: roundToInt,
			apply:      apply,
		}
		c.generators[key] = gen
		value := gen.fn(gen.params, 0)
		if roundToInt {
			value = float64(int64(value))
		}
		apply(value)
		return
	}
	delete(c.generators, key)
	apply(fm.FixedValue)
}

func (c *VLLMFakeMetricsController) startTickerLocked() {
	tickerCtx, cancel := context.WithCancel(c.rootCtx)
	c.tickerCancel = cancel
	c.tickerRunning = true
	c.tickerStart = time.Now()
	interval := c.config.FakeMetricsRefreshInterval
	go c.runTicker(tickerCtx, interval, c.tickerStart)
}

func (c *VLLMFakeMetricsController) stopTickerLocked() {
	if !c.tickerRunning {
		return
	}
	c.tickerCancel()
	c.tickerCancel = nil
	c.tickerRunning = false
}

func (c *VLLMFakeMetricsController) runTicker(ctx context.Context, interval time.Duration, start time.Time) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			c.tick(time.Since(start))
		}
	}
}

func (c *VLLMFakeMetricsController) tick(t time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, gen := range c.generators {
		value := gen.fn(gen.params, t)
		if gen.roundToInt {
			value = float64(int64(value))
		}
		gen.apply(value)
	}
}
