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

	"github.com/llm-d/llm-d-inference-sim/pkg/engine/vllm/fakemetrics"
)

// nopAdapter is the adapter NewMetricsBus falls back to when no engine
// supplied a factory. It drains every channel and exposes no metrics, which
// keeps producers from blocking on a bus nobody reads.
type nopAdapter struct{}

func (nopAdapter) Start(context.Context) error { return nil }
func (nopAdapter) Close() error                { return nil }

func (nopAdapter) OnRequestReceived(RequestReceived)         {}
func (nopAdapter) OnRequestQueued(RequestQueued)             {}
func (nopAdapter) OnRequestDequeued(RequestDequeued)         {}
func (nopAdapter) OnRequestRunning(RequestRunning)           {}
func (nopAdapter) OnPrefillStarted(PrefillStarted)           {}
func (nopAdapter) OnPrefillEnded(PrefillEnded)               {}
func (nopAdapter) OnDecodeStarted(DecodeStarted)             {}
func (nopAdapter) OnTokenGenerated(TokenGenerated)           {}
func (nopAdapter) OnDecodeEnded(DecodeEnded)                 {}
func (nopAdapter) OnRequestSucceeded(RequestSucceeded)       {}
func (nopAdapter) OnRequestFailed(RequestFailed)             {}
func (nopAdapter) OnRequestRejected(RequestRejected)         {}
func (nopAdapter) OnKVCacheUsageChanged(KVCacheUsageChanged) {}
func (nopAdapter) OnPrefixCacheQueried(PrefixCacheQueried)   {}
func (nopAdapter) OnLoRASetsChanged(LoRASetsChanged)         {}
func (nopAdapter) ApplyFakeMetricsUpdate(*fakemetrics.Config) error {
	return nil
}
