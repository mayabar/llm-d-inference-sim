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

// Unit tests for VLLMMetricsAdapter and MetricsBus that do not duplicate the
// integration coverage under pkg/tests/. These cover paths not otherwise
// exercised: MetricsBus nil-safety, adapter initial values, direct event
// handler invocations, fake-mode filtering, and ApplyUpdate scalar/reset
// paths reached without booting the simulator.

package metrics

import (
	"context"
	"time"

	"github.com/go-logr/logr"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

func newTestConfig() common.Configuration {
	return common.Configuration{
		Model:                      common.TestModelName,
		ServedModelNames:           []string{common.TestModelName},
		DisplayModelName:           common.TestModelName,
		MaxLoras:                   2,
		MaxNumSeqs:                 4,
		MaxWaitingQueueLength:      8,
		MaxModelLen:                1024,
		KVCacheSize:                128,
		TokenBlockSize:             16,
		FakeMetricsRefreshInterval: 20 * time.Millisecond,
	}
}

// newTestAdapter builds a VLLMMetricsAdapter with a fresh Prometheus registry
// and a canceled-on-cleanup context. Returned cancel stops all updater
// goroutines and the ticker.
func newTestAdapter(cfg common.Configuration) (*VLLMMetricsAdapter, *MetricsBus, context.CancelFunc) {
	ctx, cancel := context.WithCancel(context.Background())
	DeferCleanup(cancel)

	registry := prometheus.NewRegistry()
	bus, err := NewMetricsBus(ctx, cfg, registry, logr.Discard())
	Expect(err).NotTo(HaveOccurred())

	adapter, ok := bus.adapter.(*VLLMMetricsAdapter)
	Expect(ok).To(BeTrue(), "adapter is not *VLLMMetricsAdapter")
	return adapter, bus, cancel
}

// gaugeValue reads the model-labeled gauge. Adapter writes are asynchronous
// (channel -> updater goroutine -> Prometheus), so callers wrap this in
// Eventually.
func gaugeValue(g *prometheus.GaugeVec) func() float64 {
	return func() float64 {
		return testutil.ToFloat64(g.WithLabelValues(common.TestModelName))
	}
}

func counterValue(c *prometheus.CounterVec, labelValues ...string) func() float64 {
	return func() float64 {
		return testutil.ToFloat64(c.WithLabelValues(labelValues...))
	}
}

var _ = Describe("MetricsBus", func() {
	It("has nil-safe emitters", func() {
		var b *MetricsBus
		Expect(func() {
			b.EmitKVCacheUsage(0.5, false)
			b.EmitPrefixCacheQueried(1, 2)
		}).NotTo(Panic())
		Expect(b.ApplyFakeMetricsUpdate(&common.FakeMetrics{})).To(Succeed())
	})

	It("treats a nil fake-metrics update as a no-op on a live bus", func() {
		_, bus, _ := newTestAdapter(newTestConfig())
		Expect(bus.ApplyFakeMetricsUpdate(nil)).To(Succeed())
	})
})

var _ = Describe("VLLMMetricsAdapter", func() {
	Describe("initial values", func() {
		It("starts request and cache gauges at zero and stamps cache_config_info", func() {
			adapter, _, _ := newTestAdapter(newTestConfig())

			Expect(testutil.ToFloat64(adapter.runningRequests.WithLabelValues(common.TestModelName))).To(BeZero())
			Expect(testutil.ToFloat64(adapter.waitingRequests.WithLabelValues(common.TestModelName))).To(BeZero())
			Expect(testutil.ToFloat64(adapter.kvCacheUsagePercentage.WithLabelValues(common.TestModelName))).To(BeZero())
			// cache_config_info is a one-shot gauge stamped with 1 by setInitialValues.
			Expect(testutil.ToFloat64(adapter.cacheConfig.WithLabelValues("16", "128"))).To(Equal(float64(1)))
		})
	})

	Describe("event handlers", func() {
		It("updates the waiting gauge on queue/dequeue", func() {
			adapter, _, _ := newTestAdapter(newTestConfig())

			adapter.OnRequestQueued(RequestQueued{})
			adapter.OnRequestQueued(RequestQueued{})
			Eventually(gaugeValue(adapter.waitingRequests)).Should(Equal(float64(2)))

			adapter.OnRequestDequeued(RequestDequeued{QueueTime: 0.1})
			Eventually(gaugeValue(adapter.waitingRequests)).Should(Equal(float64(1)))
		})

		It("updates the running gauge on OnRequestRunning", func() {
			adapter, _, _ := newTestAdapter(newTestConfig())

			adapter.OnRequestRunning(RequestRunning{})
			adapter.OnRequestRunning(RequestRunning{})
			Eventually(gaugeValue(adapter.runningRequests)).Should(Equal(float64(2)))
		})

		It("decrements running and records tokens on success", func() {
			adapter, _, _ := newTestAdapter(newTestConfig())

			adapter.OnRequestRunning(RequestRunning{})
			Eventually(gaugeValue(adapter.runningRequests)).Should(Equal(float64(1)))

			adapter.OnRequestSucceeded(RequestSucceeded{
				PromptTokens:       10,
				GenerationTokens:   20,
				GenTokensPerChoice: []int{20},
				FinishReason:       "stop",
				E2ELatency:         0.5,
				InferenceTime:      0.4,
			})
			Eventually(gaugeValue(adapter.runningRequests)).Should(BeZero())
			Eventually(counterValue(adapter.promptTokensTotal, common.TestModelName)).Should(Equal(float64(10)))
			Eventually(counterValue(adapter.generationTokensTotal, common.TestModelName)).Should(Equal(float64(20)))
			Eventually(counterValue(adapter.requestSuccessTotal, common.TestModelName, "stop")).Should(Equal(float64(1)))
		})

		It("updates the KV-cache usage gauge", func() {
			adapter, _, _ := newTestAdapter(newTestConfig())

			adapter.OnKVCacheUsageChanged(KVCacheUsageChanged{KVCacheUsagePerc: 0.75})
			Eventually(gaugeValue(adapter.kvCacheUsagePercentage)).Should(Equal(0.75))
		})

		It("accumulates prefix-cache counters", func() {
			adapter, _, _ := newTestAdapter(newTestConfig())

			adapter.OnPrefixCacheQueried(PrefixCacheQueried{QueriedTokens: 100, CachedPromptTokens: 40})
			adapter.OnPrefixCacheQueried(PrefixCacheQueried{QueriedTokens: 50, CachedPromptTokens: 10})
			Eventually(counterValue(adapter.prefixCacheQueriesTotal, common.TestModelName)).Should(Equal(float64(150)))
			Eventually(counterValue(adapter.prefixCacheHitsTotal, common.TestModelName)).Should(Equal(float64(50)))
		})
	})

	Describe("fake mode", func() {
		It("drops real-path events", func() {
			cfg := newTestConfig()
			cfg.FakeMetrics = &common.FakeMetrics{}
			adapter, _, _ := newTestAdapter(cfg)

			adapter.OnRequestQueued(RequestQueued{})
			adapter.OnRequestRunning(RequestRunning{})
			adapter.OnKVCacheUsageChanged(KVCacheUsageChanged{KVCacheUsagePerc: 0.9})
			adapter.OnPrefixCacheQueried(PrefixCacheQueried{QueriedTokens: 5, CachedPromptTokens: 3})

			// Nothing should change; Consistently gives the updater goroutines a
			// chance to run and still find zeros.
			Consistently(gaugeValue(adapter.waitingRequests), 50*time.Millisecond).Should(BeZero())
			Expect(gaugeValue(adapter.runningRequests)()).To(BeZero())
			Expect(gaugeValue(adapter.kvCacheUsagePercentage)()).To(BeZero())
			Expect(counterValue(adapter.prefixCacheQueriesTotal, common.TestModelName)()).To(BeZero())
		})

		It("applies fixed scalar values via ApplyUpdate", func() {
			cfg := newTestConfig()
			cfg.FakeMetrics = &common.FakeMetrics{}
			adapter, _, _ := newTestAdapter(cfg)

			kv := 0.4
			upd := &common.FakeMetrics{
				RunningRequests:        &common.FakeMetricWithFunction{FixedValue: 3},
				WaitingRequests:        &common.FakeMetricWithFunction{FixedValue: 7},
				KVCacheUsagePercentage: &common.FakeMetricWithFunction{FixedValue: kv},
			}
			Expect(adapter.ApplyUpdate(upd)).To(Succeed())

			Eventually(gaugeValue(adapter.runningRequests)).Should(Equal(float64(3)))
			Eventually(gaugeValue(adapter.waitingRequests)).Should(Equal(float64(7)))
			Eventually(gaugeValue(adapter.kvCacheUsagePercentage)).Should(Equal(kv))
		})

		It("starts and stops the ticker as generators are enabled and cleared", func() {
			cfg := newTestConfig()
			cfg.FakeMetrics = &common.FakeMetrics{}
			adapter, _, _ := newTestAdapter(cfg)

			// Simulate that Start() has already run — the ticker only auto-starts
			// once the adapter is marked started.
			adapter.genMu.Lock()
			adapter.started = true
			adapter.genMu.Unlock()

			// Enable a generator: ticker must come up.
			upd := &common.FakeMetrics{
				RunningRequests: &common.FakeMetricWithFunction{
					IsFunction: true,
					Function: &common.FunctionInfo{
						Name:   common.RampFuncName,
						Start:  0,
						End:    5,
						Period: 100 * time.Millisecond,
					},
				},
			}
			Expect(adapter.ApplyUpdate(upd)).To(Succeed())

			adapter.genMu.Lock()
			running := adapter.tickerRunning
			adapter.genMu.Unlock()
			Expect(running).To(BeTrue(), "expected ticker running after generator enabled")

			// Replace the generator with a fixed value: generator set becomes empty,
			// ticker must stop.
			upd2 := &common.FakeMetrics{
				RunningRequests: &common.FakeMetricWithFunction{FixedValue: 2},
			}
			Expect(adapter.ApplyUpdate(upd2)).To(Succeed())

			adapter.genMu.Lock()
			running = adapter.tickerRunning
			adapter.genMu.Unlock()
			Expect(running).To(BeFalse(), "expected ticker stopped after generators cleared")
		})

		It("stops the ticker on Close", func() {
			cfg := newTestConfig()
			cfg.FakeMetrics = &common.FakeMetrics{}
			adapter, _, _ := newTestAdapter(cfg)

			adapter.genMu.Lock()
			adapter.started = true
			adapter.genMu.Unlock()

			upd := &common.FakeMetrics{
				WaitingRequests: &common.FakeMetricWithFunction{
					IsFunction: true,
					Function: &common.FunctionInfo{
						Name:   common.RampFuncName,
						Start:  0,
						End:    10,
						Period: 100 * time.Millisecond,
					},
				},
			}
			Expect(adapter.ApplyUpdate(upd)).To(Succeed())
			Expect(adapter.Close()).To(Succeed())

			adapter.genMu.Lock()
			running := adapter.tickerRunning
			started := adapter.started
			adapter.genMu.Unlock()
			Expect(running).To(BeFalse(), "expected ticker stopped after Close")
			Expect(started).To(BeFalse(), "expected started=false after Close")
		})
	})

	Describe("LoRA ref counting", func() {
		It("moves adapters through waiting -> running -> done", func() {
			adapter, _, _ := newTestAdapter(newTestConfig())

			// The lorasUpdater goroutine serializes all state changes, so we poll
			// for the final state after each phase.
			adapter.writeToLoRAs(LoRAUpdate{Usage: &loraUsage{name: "a", state: waitingUsageState}})
			adapter.writeToLoRAs(LoRAUpdate{Usage: &loraUsage{name: "b", state: waitingUsageState}})
			Eventually(loraKeys(&adapter.waitingLoras)).Should(ConsistOf("a", "b"))

			adapter.writeToLoRAs(LoRAUpdate{Usage: &loraUsage{name: "a", state: runningUsageState}})
			Eventually(loraKeys(&adapter.waitingLoras)).Should(ConsistOf("b"))
			Eventually(loraKeys(&adapter.runningLoras)).Should(ConsistOf("a"))

			adapter.writeToLoRAs(LoRAUpdate{Usage: &loraUsage{name: "a", state: doneUsageState}})
			adapter.writeToLoRAs(LoRAUpdate{Usage: &loraUsage{name: "b", state: runningUsageState}})
			adapter.writeToLoRAs(LoRAUpdate{Usage: &loraUsage{name: "b", state: doneUsageState}})
			Eventually(loraKeys(&adapter.waitingLoras)).Should(BeEmpty())
			Eventually(loraKeys(&adapter.runningLoras)).Should(BeEmpty())
		})
	})
})

// loraKeys returns a poller that snapshots the string keys of a sync.Map-like
// container (any type implementing Range).
func loraKeys(m interface {
	Range(func(any, any) bool)
}) func() []string {
	return func() []string {
		var got []string
		m.Range(func(k, _ any) bool {
			if s, ok := k.(string); ok {
				got = append(got, s)
			}
			return true
		})
		return got
	}
}
