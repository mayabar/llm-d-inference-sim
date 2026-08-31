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
	"math"
	"testing"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

func params(start, end float64, period time.Duration) *common.FunctionInfo {
	return &common.FunctionInfo{Start: start, End: end, Period: period}
}

func nearlyEqual(a, b float64) bool {
	return math.Abs(a-b) < 1e-9
}

func TestOscillate(t *testing.T) {
	p := params(0, 10, time.Second)
	// t=0 -> mid + amp*sin(0) = 5
	if v := Oscillate(p, 0); !nearlyEqual(v, 5) {
		t.Errorf("Oscillate t=0 got %v want 5", v)
	}
	// t=period/4 -> mid + amp*sin(pi/2) = 5 + 5 = 10
	if v := Oscillate(p, 250*time.Millisecond); !nearlyEqual(v, 10) {
		t.Errorf("Oscillate t=period/4 got %v want 10", v)
	}
	// t=period/2 -> mid + amp*sin(pi) = 5
	if v := Oscillate(p, 500*time.Millisecond); !nearlyEqual(v, 5) {
		t.Errorf("Oscillate t=period/2 got %v want 5", v)
	}
}

func TestRamp(t *testing.T) {
	p := params(0, 10, time.Second)
	if v := Ramp(p, 0); !nearlyEqual(v, 0) {
		t.Errorf("Ramp t=0 got %v want 0", v)
	}
	if v := Ramp(p, 500*time.Millisecond); !nearlyEqual(v, 5) {
		t.Errorf("Ramp t=period/2 got %v want 5", v)
	}
	if v := Ramp(p, time.Second); !nearlyEqual(v, 10) {
		t.Errorf("Ramp t=period got %v want 10", v)
	}
	// past period: stays at End
	if v := Ramp(p, 3*time.Second); !nearlyEqual(v, 10) {
		t.Errorf("Ramp t=3period got %v want 10", v)
	}
}

func TestRampWithReset(t *testing.T) {
	p := params(0, 10, time.Second)
	if v := RampWithReset(p, 0); !nearlyEqual(v, 0) {
		t.Errorf("RampWithReset t=0 got %v want 0", v)
	}
	if v := RampWithReset(p, 500*time.Millisecond); !nearlyEqual(v, 5) {
		t.Errorf("RampWithReset t=period/2 got %v want 5", v)
	}
	// wraps back to Start at t=period
	if v := RampWithReset(p, time.Second); !nearlyEqual(v, 0) {
		t.Errorf("RampWithReset t=period got %v want 0", v)
	}
	// wraps in the middle of the second period
	if v := RampWithReset(p, 1500*time.Millisecond); !nearlyEqual(v, 5) {
		t.Errorf("RampWithReset t=1.5period got %v want 5", v)
	}
}

func TestSquarewave(t *testing.T) {
	p := params(2, 8, time.Second)
	if v := Squarewave(p, 0); !nearlyEqual(v, 2) {
		t.Errorf("Squarewave t=0 got %v want 2", v)
	}
	if v := Squarewave(p, 250*time.Millisecond); !nearlyEqual(v, 2) {
		t.Errorf("Squarewave t=period/4 got %v want 2", v)
	}
	if v := Squarewave(p, 500*time.Millisecond); !nearlyEqual(v, 8) {
		t.Errorf("Squarewave t=period/2 got %v want 8", v)
	}
	if v := Squarewave(p, 750*time.Millisecond); !nearlyEqual(v, 8) {
		t.Errorf("Squarewave t=3period/4 got %v want 8", v)
	}
	// wraps to Start at the next period
	if v := Squarewave(p, time.Second); !nearlyEqual(v, 2) {
		t.Errorf("Squarewave t=period got %v want 2", v)
	}
}

func TestDispatch(t *testing.T) {
	cases := map[string]Generator{
		common.OscillateFuncName:     Oscillate,
		common.RampFuncName:          Ramp,
		common.RampWithResetFuncName: RampWithReset,
		common.SquarewaveFuncName:    Squarewave,
	}
	p := params(0, 10, time.Second)
	for name, want := range cases {
		got := Dispatch(name)
		if got == nil {
			t.Errorf("Dispatch(%q) returned nil", name)
			continue
		}
		// Compare by evaluated output at a sampling point rather than function
		// pointer identity — Go does not guarantee identity for named
		// functions.
		at := 250 * time.Millisecond
		if got(p, at) != want(p, at) {
			t.Errorf("Dispatch(%q) returned unexpected function", name)
		}
	}
	if Dispatch("nope") != nil {
		t.Error("Dispatch of unknown name should be nil")
	}
}
