package main

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

const (
	seed = 1234567
)

func oneThreadTestSequence(testIndex, numOfTests, numOfIterations, delayBetweenCalls int) {
	for testSubIndex := range numOfTests {
		oneThreadTest(testIndex, testSubIndex, numOfIterations, delayBetweenCalls)
	}
}

// numOfIterations - number of iterations
// delayBetweenCalls - delay between random value generation calls
// prints input + netto time of random numbers generation
func oneThreadTest(testIndex, testSubIndex, numOfIterations, delayBetweenCalls int) {
	// fmt.Printf("oneThreadTest: %d, %d\n", testIndex, testSubIndex)
	var randMutex sync.Mutex
	src := rand.NewSource(seed)
	randomGenerator := rand.New(src)

	startTime := time.Now()

	for range numOfIterations {
		randMutex.Lock()
		randomGenerator.Intn(100)
		randMutex.Unlock()
		if delayBetweenCalls > 0 {
			time.Sleep(time.Duration(delayBetweenCalls * int(time.Millisecond)))
		}
	}
	elapsed := time.Since(startTime)
	printResult(testIndex, testSubIndex, true, false, numOfIterations, elapsed-(time.Duration(numOfIterations*delayBetweenCalls*int(time.Millisecond))), elapsed, delayBetweenCalls)

	startTime = time.Now()

	for range numOfIterations {
		rand.New(rand.NewSource(time.Now().UnixNano())).Intn(100)
		if delayBetweenCalls > 0 {
			time.Sleep(time.Duration(delayBetweenCalls * int(time.Millisecond)))
		}
	}
	elapsed = time.Since(startTime)

	printResult(testIndex, testSubIndex, false, false, numOfIterations, elapsed-(time.Duration(numOfIterations*delayBetweenCalls*int(time.Millisecond))), elapsed, delayBetweenCalls)

}

func durationToMillis(d time.Duration) float64 {
	return float64(d) / float64(time.Millisecond)
}
func printResult(testIndex, testSubIndex int, withMutex bool, isMultiThread bool, numOfIterations int, workTime time.Duration, totalTime time.Duration, delayBetweenCalls int) {
	workTimeMs := durationToMillis(workTime)

	fmt.Printf("%d, %d, %t, %t, %d, %d, %.4f, %.4f, %.6f\n", testIndex, testSubIndex, withMutex, isMultiThread, numOfIterations, delayBetweenCalls, durationToMillis(totalTime), workTimeMs, workTimeMs/float64(numOfIterations))
}

func printHeader() {
	fmt.Println("test index, test sub index, uses mutex, is multi thread, # iterations, delay between calls (ms), total time (ms), work time (ms), average time per call (ms)")
}

func main() {
	printHeader()
	// oneThreadTestSequence(1, 2, 100000, 100)
	oneThreadTestSequence(1, 2, 100000, 10)
	oneThreadTestSequence(2, 2, 100000, 0)
}
