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

import "math"

// Build125Buckets generates histogram buckets in powers of 10 scaled by [1,2,5].
// This matches vLLM's build_1_2_5_buckets() in metrics.py.
//
// Reference: https://github.com/vllm-project/vllm/blob/main/vllm/engine/metrics.py#L175
func Build125Buckets(maxValue int) []float64 {
	if maxValue <= 0 {
		return []float64{}
	}
	var buckets []float64
	exponent := 0
	mantissa := []int{1, 2, 5}

	for {
		complete := true
		for _, m := range mantissa {
			value := m * int(math.Pow10(exponent))
			if value <= maxValue {
				buckets = append(buckets, float64(value))
				complete = false
			}
		}
		if complete {
			break
		}
		exponent++
	}
	return buckets
}
