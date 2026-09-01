/*
Copyright 2026 The llm-d-inference-simference-sim Authors.

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

// Contains functions related to fake prometheus metrics

package simulator

import (
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/metrics"
)

// UpdateFakeMetricsFromBody applies a partial fake-metrics update parsed from
// the body of the deprecated /fake_metrics endpoint. The body is the partial
// itself (not wrapped in {"fake-metrics": ...}); we wrap it and dispatch
// through ApplyConfigUpdate so it goes through the same validation, atomic
// config swap, and Prometheus side-effect path as /admin/config.
// Will be removed in v0.12.
func (s *SimContext) UpdateFakeMetricsFromBody(body []byte) error {
	wrapped := append(append([]byte(`{"fake-metrics":`), body...), '}')
	return s.ApplyConfigUpdate(wrapped)
}

type generatedFakeMetrics struct {
	updateChan common.Channel[common.MetricInfo]
	genFun     metrics.Generator
	params     *common.FunctionInfo
	roundToInt bool
}

func (s *SimContext) setInitialFakeMetrics() error {
	s.metrics.generatedFakeMetrics = make(map[string]generatedFakeMetrics)

	initial := s.Config().FakeMetrics

	// Loras always need processing on initial setup so the default empty
	// entry (no adapters, current timestamp) gets registered. Parser
	// initializes LoraMetrics to a non-nil (possibly empty) slice for the
	// configured case; force non-nil here to cover any path that didn't.
	if initial.LoraMetrics == nil {
		initial.LoraMetrics = []common.LorasMetrics{}
	}
	return s.updateFakeMetrics(initial, nil)
}

func (s *SimContext) updateGeneratedFakeMetrics() {
	start := time.Now()
	ticker := time.NewTicker(s.Config().FakeMetricsRefreshInterval)
	defer ticker.Stop()
	for {
		select {
		case <-s.metrics.stopFakeMetricsTicker:
			return
		case <-ticker.C:
			t := time.Since(start)
			for _, metric := range s.metrics.generatedFakeMetrics {
				value := metric.genFun(metric.params, t)
				if metric.roundToInt {
					rounded := int64(value)
					value = float64(rounded)
				}
				update := common.MetricInfo{
					Value:  value,
					IsFake: true,
				}
				common.WriteToChannel(metric.updateChan, update, s.logger)
			}
		}
	}
}

// updateFakeMetrics applies a partial update to the simulator's Prometheus
// metrics. update carries the fields to apply (nil fields are skipped); old
// is the previous FakeMetrics state used to decide whether a collector
// already exists and must be unregistered+recreated to drop accumulated
// observations. old is nil at startup (setInitialFakeMetrics) where there is
// no prior state.
//
// This function does not mutate any shared state — the merged FakeMetrics is
// produced by Configuration.Update and swapped in by the caller via
// SetConfig.
func (s *SimContext) updateFakeMetrics(update *common.FakeMetrics, old *common.FakeMetrics) error {
	var generatedFakeMetricsWasEmpty bool
	if len(s.metrics.generatedFakeMetrics) == 0 {
		generatedFakeMetricsWasEmpty = true
	}

	if update.RunningRequests != nil {
		s.setFakeMetricWithFunction(s.Config().DisplayModelName, update.RunningRequests, s.metrics.runningRequests,
			s.metrics.runReqChan, true)
	}
	if update.WaitingRequests != nil {
		s.setFakeMetricWithFunction(s.Config().DisplayModelName, update.WaitingRequests, s.metrics.waitingRequests,
			s.metrics.waitingReqChan, true)
	}
	if update.KVCacheUsagePercentage != nil {
		s.setFakeMetricWithFunction(s.Config().DisplayModelName, update.KVCacheUsagePercentage, s.metrics.kvCacheUsagePercentage,
			s.metrics.kvCacheUsageChan, false)
	}

	if update.TTFTBucketValues != nil {
		if old != nil && old.TTFTBucketValues != nil {
			s.metrics.registry.Unregister(s.metrics.ttft)
			if err := s.createAndRegisterTTFTMetric(); err != nil {
				return err
			}
		}
		metrics.InitFakeHistogram(s.metrics.ttft, s.Config().DisplayModelName, common.TTFTBucketsBoundaries, update.TTFTBucketValues)
	}

	if update.TPOTBucketValues != nil {
		if old != nil && old.TPOTBucketValues != nil {
			s.metrics.registry.Unregister(s.metrics.tpot)
			s.metrics.registry.Unregister(s.metrics.interTokenLatency)
			if err := s.createAndRegisterTPOTAndInterTokenMetrics(); err != nil {
				return err
			}
		}
		metrics.InitFakeHistogram(s.metrics.tpot, s.Config().DisplayModelName, common.TPOTBucketsBoundaries, update.TPOTBucketValues)
		metrics.InitFakeHistogram(s.metrics.interTokenLatency, s.Config().DisplayModelName, common.TPOTBucketsBoundaries, update.TPOTBucketValues)
	}

	if update.E2ERequestLatencyBucketValues != nil {
		if old != nil && old.E2ERequestLatencyBucketValues != nil {
			s.metrics.registry.Unregister(s.metrics.e2eReqLatency)
			if err := s.createAndRegisterE2EReqLatencyMetric(); err != nil {
				return err
			}
		}
		metrics.InitFakeHistogram(s.metrics.e2eReqLatency, s.Config().DisplayModelName, common.RequestLatencyBucketsBoundaries, update.E2ERequestLatencyBucketValues)
	}

	if update.ReqQueueTimeBucketValues != nil {
		if old != nil && old.ReqQueueTimeBucketValues != nil {
			s.metrics.registry.Unregister(s.metrics.reqQueueTime)
			if err := s.createAndRegisterReqQueueTimeMetric(); err != nil {
				return err
			}
		}
		metrics.InitFakeHistogram(s.metrics.reqQueueTime, s.Config().DisplayModelName, common.RequestLatencyBucketsBoundaries, update.ReqQueueTimeBucketValues)
	}

	if update.ReqInfTimeBucketValues != nil {
		if old != nil && old.ReqInfTimeBucketValues != nil {
			s.metrics.registry.Unregister(s.metrics.reqInferenceTime)
			if err := s.createAndRegisterReqInferenceTimeMetric(); err != nil {
				return err
			}
		}
		metrics.InitFakeHistogram(s.metrics.reqInferenceTime, s.Config().DisplayModelName, common.RequestLatencyBucketsBoundaries, update.ReqInfTimeBucketValues)
	}

	if update.ReqPrefillTimeBucketValues != nil {
		if old != nil && old.ReqPrefillTimeBucketValues != nil {
			s.metrics.registry.Unregister(s.metrics.reqPrefillTime)
			if err := s.createAndRegisterReqPrefillTimeMetric(); err != nil {
				return err
			}
		}
		metrics.InitFakeHistogram(s.metrics.reqPrefillTime, s.Config().DisplayModelName, common.RequestLatencyBucketsBoundaries, update.ReqPrefillTimeBucketValues)
	}

	if update.ReqDecodeTimeBucketValues != nil {
		if old != nil && old.ReqDecodeTimeBucketValues != nil {
			s.metrics.registry.Unregister(s.metrics.reqDecodeTime)
			if err := s.createAndRegisterReqDecodeTimeMetric(); err != nil {
				return err
			}
		}
		metrics.InitFakeHistogram(s.metrics.reqDecodeTime, s.Config().DisplayModelName, common.RequestLatencyBucketsBoundaries, update.ReqDecodeTimeBucketValues)
	}

	if update.ReqTPOTBucketValues != nil {
		if old != nil && old.ReqTPOTBucketValues != nil {
			s.metrics.registry.Unregister(s.metrics.reqTpot)
			if err := s.createAndRegisterReqTpotMetric(); err != nil {
				return err
			}
		}
		metrics.InitFakeHistogram(s.metrics.reqTpot, s.Config().DisplayModelName, common.TPOTBucketsBoundaries, update.ReqTPOTBucketValues)
	}

	buckets := metrics.Build125Buckets(s.Config().MaxModelLen)

	if update.RequestParamsMaxTokens != nil {
		if old != nil && old.RequestParamsMaxTokens != nil {
			s.metrics.registry.Unregister(s.metrics.requestParamsMaxTokens)
			if err := s.createAndRegisterReqParamsMaxTokensMetric(); err != nil {
				return err
			}
		}
		metrics.InitFakeHistogram(s.metrics.requestParamsMaxTokens, s.Config().DisplayModelName, buckets, update.RequestParamsMaxTokens)
	}

	if update.RequestMaxGenerationTokens != nil {
		if old != nil && old.RequestMaxGenerationTokens != nil {
			s.metrics.registry.Unregister(s.metrics.maxNumGenerationTokens)
			if err := s.createAndRegisterMaxNumGenerationTokensMetric(); err != nil {
				return err
			}
		}
		metrics.InitFakeHistogram(s.metrics.maxNumGenerationTokens, s.Config().DisplayModelName, buckets, update.RequestMaxGenerationTokens)
	}

	var oldRequestPromptTokens, oldRequestGenerationTokens []int
	var oldTotalPromptTokens, oldTotalGenerationTokens *int64
	var oldPrefixCacheQueries, oldPrefixCacheHits *int64
	var oldRequestSuccessTotal map[string]int64
	if old != nil {
		oldRequestPromptTokens = old.RequestPromptTokens
		oldRequestGenerationTokens = old.RequestGenerationTokens
		oldTotalPromptTokens = old.TotalPromptTokens
		oldTotalGenerationTokens = old.TotalGenerationTokens
		oldPrefixCacheQueries = old.PrefixCacheQueries
		oldPrefixCacheHits = old.PrefixCacheHits
		oldRequestSuccessTotal = old.RequestSuccessTotal
	}

	if err := s.updateTokenMetrics(
		s.Config().DisplayModelName, buckets,
		update.RequestPromptTokens, oldRequestPromptTokens,
		update.TotalPromptTokens, oldTotalPromptTokens,
		&s.metrics.requestPromptTokens, &s.metrics.promptTokensTotal,
		s.createAndRegisterReqPromptTokensMetrics, s.createAndRegisterPromptTokensTotalMetrics,
	); err != nil {
		return err
	}

	if err := s.updateTokenMetrics(
		s.Config().DisplayModelName, buckets,
		update.RequestGenerationTokens, oldRequestGenerationTokens,
		update.TotalGenerationTokens, oldTotalGenerationTokens,
		&s.metrics.requestGenerationTokens, &s.metrics.generationTokensTotal,
		s.createAndRegisterReqGenerationTokensMetrics, s.createAndRegisterGenerationTokensTotalMetrics,
	); err != nil {
		return err
	}

	if update.PrefixCacheQueries != nil {
		if oldPrefixCacheQueries != nil {
			s.metrics.registry.Unregister(s.metrics.prefixCacheQueriesTotal)
			if err := s.createAndRegisterPrefixCacheQueriesTotalMetric(); err != nil {
				return err
			}
		}
		s.metrics.prefixCacheQueriesTotal.WithLabelValues(s.Config().DisplayModelName).Add(float64(*update.PrefixCacheQueries))
	}

	if update.PrefixCacheHits != nil {
		if oldPrefixCacheHits != nil {
			s.metrics.registry.Unregister(s.metrics.prefixCacheHitsTotal)
			if err := s.createAndRegisterPrefixCacheHitsTotalMetric(); err != nil {
				return err
			}
		}
		s.metrics.prefixCacheHitsTotal.WithLabelValues(s.Config().DisplayModelName).Add(float64(*update.PrefixCacheHits))
	}

	if update.RequestSuccessTotal != nil {
		if oldRequestSuccessTotal != nil {
			s.metrics.registry.Unregister(s.metrics.requestSuccessTotal)
			if err := s.createAndRegisterRequestSuccessTotalMetric(); err != nil {
				return err
			}
		}
		for reason, requestSuccessTotal := range update.RequestSuccessTotal {
			s.metrics.requestSuccessTotal.WithLabelValues(s.Config().DisplayModelName, reason).Add(float64(requestSuccessTotal))
		}
	}

	if update.LoraMetrics != nil {
		s.metrics.registry.Unregister(s.metrics.loraInfo)
		if err := s.createAndRegisterLoraInfoMetric(); err != nil {
			return err
		}
		if len(update.LoraMetrics) != 0 {
			for _, metrics := range update.LoraMetrics {
				s.metrics.loraInfo.WithLabelValues(
					strconv.Itoa(s.Config().MaxLoras),
					metrics.RunningLoras,
					metrics.WaitingLoras).Set(metrics.Timestamp)
			}
		} else {
			s.metrics.loraInfo.WithLabelValues(
				strconv.Itoa(s.Config().MaxLoras),
				"",
				"").Set(float64(time.Now().Unix()))
		}
	}

	if generatedFakeMetricsWasEmpty && len(s.metrics.generatedFakeMetrics) > 0 {
		s.metrics.stopFakeMetricsTicker = make(chan struct{})
		go s.updateGeneratedFakeMetrics()
	} else if !generatedFakeMetricsWasEmpty && len(s.metrics.generatedFakeMetrics) == 0 {
		close(s.metrics.stopFakeMetricsTicker)
	}

	return nil
}

// updateTokenMetrics handles the update logic for a histogram+counter token
// metric pair. It updates the histogram if new values are provided, then
// conditionally resets and updates the associated total counter based on what
// changed between old and new configurations. No state is mutated; the merged
// FakeMetrics is produced separately by Configuration.Update.
func (s *SimContext) updateTokenMetrics(
	modelName string,
	buckets []float64,
	newHistValues []int,
	oldHistValues []int,
	newExplicitTotal *int64,
	oldExplicitTotal *int64,
	hist **prometheus.HistogramVec,
	counter **prometheus.CounterVec,
	recreateHist func() error,
	recreateCounter func() error,
) error {
	newHasHist := newHistValues != nil
	newHasExplicit := newExplicitTotal != nil

	// Update histogram if new values are provided.
	var histTotal *int64
	if newHasHist {
		if oldHistValues != nil {
			s.metrics.registry.Unregister(*hist)
			if err := recreateHist(); err != nil {
				return err
			}
		}
		histTotal = metrics.InitFakeHistogram(*hist, modelName, buckets, newHistValues)
	}

	// The counter can be set from two sources: an explicit total value,
	// or derived from the request histogram.
	if !newHasHist && !newHasExplicit {
		return nil
	}

	// Reset (unregister + re-register) if the counter already had a value.
	if oldExplicitTotal != nil || oldHistValues != nil {
		s.metrics.registry.Unregister(*counter)
		if err := recreateCounter(); err != nil {
			return err
		}
	}

	// Use the explicit total if provided, otherwise use the total derived from
	// the histogram.
	tokenTotal := histTotal
	if newHasExplicit {
		tokenTotal = newExplicitTotal
	}
	if tokenTotal != nil {
		(*counter).WithLabelValues(modelName).Add(float64(*tokenTotal))
	}

	return nil
}

func (s *SimContext) setFakeMetricWithFunction(modelName string, fm *common.FakeMetricWithFunction, metric *prometheus.GaugeVec,
	channel common.Channel[common.MetricInfo], roundToInt bool) {
	var value float64
	if fm.IsFunction {
		genFakeMetric := generatedFakeMetrics{
			updateChan: channel,
			genFun:     metrics.Dispatch(fm.Function.Name),
			params:     fm.Function,
			roundToInt: roundToInt,
		}
		s.metrics.generatedFakeMetrics[channel.Name] = genFakeMetric
		value = genFakeMetric.genFun(genFakeMetric.params, 0)
	} else {
		delete(s.metrics.generatedFakeMetrics, channel.Name)
		value = fm.FixedValue
	}
	metric.WithLabelValues(modelName).Set(value)
}
