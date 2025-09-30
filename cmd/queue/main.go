package main

import (
	"container/list"
	"context"
	"errors"
	"strconv"
	"sync"
	"time"

	"k8s.io/klog/v2"
)

var ErrNoFreeWorker = errors.New("no free worker available")
var ErrQueueFull = errors.New("waiting queue full")

var wg sync.WaitGroup
var logger = klog.Background()

type Worker struct {
	id int
}

func (w *Worker) ProcessRequest(req *Request) error {
	logger.Info("start processing", "req", req.id, "worker", w.id)
	time.Sleep(2000 * time.Millisecond)
	logger.Info("finish processing", "req", req.id, "worker", w.id)
	wg.Done()
	return nil
}

// //////////////////////////////////////////////////////////
type Request struct {
	id    string
	model string
}

func (r *Request) GetModel() string {
	return r.model
}

type RequestWorkerPair struct {
	request *Request
	worker  *Worker
}

// //////////////////////////////////////////////////////////
type RequestManager struct {
	// workers pool
	freeWorkers chan *Worker

	// lora references data
	maxLora      int
	usedLorasMux sync.RWMutex
	// lora adapter name -> reference count (number of currently running requests)
	usedLoras map[string]int

	// waiting requests queue
	mux sync.Mutex
	// bi-directional list of *Request
	waitingQueue  *list.List
	queueCapacity int

	ctx context.Context

	// request-workers channel
	reqsProcessingChan chan *RequestWorkerPair
}

// NewRequestManager creates a new RequestManager instance
func NewRequestManager(numWorkers int, maxConcurentRequests int, maxLora int, waitingQueueSize int) *RequestManager {
	rm := &RequestManager{
		freeWorkers:        make(chan *Worker, numWorkers),
		maxLora:            maxLora,
		usedLoras:          make(map[string]int),
		queueCapacity:      waitingQueueSize,
		waitingQueue:       list.New(),
		reqsProcessingChan: make(chan *RequestWorkerPair, numWorkers),
	}

	for i := range numWorkers {
		worker := &Worker{id: i}
		// initially all workers are free
		rm.freeWorkers <- worker
	}

	return rm
}

func (rm *RequestManager) Start(ctx context.Context) {
	rm.ctx = ctx

	go rm.processing()
}

// GetFreeWorker returns a free worker or error if none are available (non-blocking)
func (rm *RequestManager) GetFreeWorker() (*Worker, error) {
	select {
	case w := <-rm.freeWorkers:
		// logger.Info("GetFreeWorker", "worker", w.id)
		return w, nil
	default:
		// logger.Info("GetFreeWorker", "worker", "null")
		return nil, ErrNoFreeWorker
	}
}

// SubmitRequest attempts to process or enqueue the incoming request
func (rm *RequestManager) HandleRequest(req *Request) error {
	// logger.Info("HandleRequest", "request", req.id)

	// check if lora usage allows request to run
	if !rm.canUseLora(req.GetModel()) {
		logger.Info("Lora cannot be loaded - send request to waiting queue", "lora", req.GetModel())
		// LoRA max reached, try to enqueue
		if err := rm.Enqueue(req); err != nil {
			return errors.Join(ErrQueueFull, err)
		}
		return nil
	}

	// from lora's point of view this request can run - try to get a free worker
	worker, err := rm.GetFreeWorker()
	if err != nil {
		// logger.Info("No free worker - send request to the waiting queue")
		// no free worker, add this request to the waiting queue
		if err := rm.Enqueue(req); err != nil {
			return errors.Join(ErrQueueFull, err)
		}
		return nil
	}

	// go rm.processRequest(worker, req) // asynchronously process
	// logger.Info("Send request to processing channel", "req", req.id, "worker", worker.id)
	rm.reqsProcessingChan <- &RequestWorkerPair{request: req, worker: worker}

	return nil
}

func (rm *RequestManager) processing() {
	// logger.Info("Start processing routine")

	for {
		select {
		case <-rm.ctx.Done():
			logger.Info("Done")
			return
		case worker := <-rm.freeWorkers:
			// logger.Info("Free worker", "worker", worker)
			// there is a free worker - find a request for it
			nextReq := rm.Dequeue()
			if nextReq != nil {
				// send this request for processing in this worker
				// logger.Info("Free worker and request were found - send to processing", "req", nextReq.id, "worker", worker.id)
				rm.reqsProcessingChan <- &RequestWorkerPair{request: nextReq, worker: worker}
			} else {
				// no waiting request, return worker to be free
				rm.freeWorkers <- worker
			}
		case rpPair := <-rm.reqsProcessingChan:
			// logger.Info("Got a pair from the request processing channel", "req", rpPair.request.id, "worker", rpPair.worker.id)
			go func() {
				// logger.Info("Increment lora\n")
				// this request can run + there is a free worker - assign request to worker and update LoRA usage
				rm.incrementLora(rpPair.request.GetModel())

				err := rpPair.worker.ProcessRequest(rpPair.request)
				if err != nil {
					logger.Info("Worker failed to process request", "worker", rpPair.worker.id, "reqeust", rpPair.request.id)
				}

				// decrement the LoRA usage
				rm.decrementLora(rpPair.request.GetModel())
				// return the worker to the list of free workers
				rm.freeWorkers <- rpPair.worker
			}()
		}
	}
}

// Checks if a request with this model can run under maxLora limit
func (rm *RequestManager) canUseLora(model string) bool {
	// TODO use simulator.isLora
	if model == "base" {
		return true
	}

	rm.usedLorasMux.RLock()
	defer rm.usedLorasMux.RUnlock()
	// check if this lora is already loaded or within maxLora slots
	return (rm.usedLoras[model] > 0 || len(rm.usedLoras) < rm.maxLora)
}

func (rm *RequestManager) incrementLora(model string) {
	if model == "base" {
		return
	}

	rm.usedLorasMux.Lock()
	defer rm.usedLorasMux.Unlock()
	rm.usedLoras[model]++
}

func (rm *RequestManager) decrementLora(model string) {
	if model == "base" {
		return
	}

	rm.usedLorasMux.Lock()
	defer rm.usedLorasMux.Unlock()

	if rm.usedLoras[model] <= 1 {
		// last usage of this lora - remove it from the loaded loras list
		delete(rm.usedLoras, model)
	} else {
		rm.usedLoras[model] -= 1
	}
}

func (rm *RequestManager) Enqueue(req *Request) error {
	// logger.Info("Enqueue", "req", req)

	rm.mux.Lock()
	defer rm.mux.Unlock()

	if rm.waitingQueue.Len() >= rm.queueCapacity {
		return ErrQueueFull
	}
	rm.waitingQueue.PushBack(req)
	return nil
}

// go though the queue and find the first request that can be executed, while taking into consideration the max lora limitation
func (rm *RequestManager) Dequeue() *Request {
	rm.mux.Lock()
	defer rm.mux.Unlock()

	for elem := rm.waitingQueue.Front(); elem != nil; elem = elem.Next() {
		req, ok := elem.Value.(*Request)
		if ok && req != nil && rm.canUseLora(req.GetModel()) {
			// found the request that can be processed from the lora point of view
			// logger.Info("Dequeue element", "req", req)
			rm.waitingQueue.Remove(elem)
			return req
		}
	}

	return nil
}

type MyTest struct {
	delayMillis int
	model       string
}

func main() {
	logger.Info("Start")
	reqMgr := NewRequestManager(5, 5, 2, 20)
	reqMgr.Start(context.Background())

	// tests := []MyTest{{
	// 	delayMillis: 0,
	// 	model:       "l1",
	// }, {
	// 	delayMillis: 100,
	// 	model:       "l2",
	// }, {
	// 	delayMillis: 200,
	// 	model:       "l3",
	// }, {
	// 	delayMillis: 300,
	// 	model:       "l4",
	// }, {
	// 	delayMillis: 400,
	// 	model:       "l1",
	// }, {
	// 	delayMillis: 500,
	// 	model:       "l1",
	// }}

	tests := []MyTest{{
		delayMillis: 0,
		model:       "l1",
	}, {
		delayMillis: 0,
		model:       "l2",
	}, {
		delayMillis: 0,
		model:       "l3",
	}, {
		delayMillis: 0,
		model:       "l4",
	}, {
		delayMillis: 0,
		model:       "l1",
	}, {
		delayMillis: 0,
		model:       "l1",
	}}

	wg.Add(len(tests))

	for i, test := range tests {
		go func() {
			time.Sleep(time.Duration(test.delayMillis) * time.Millisecond)
			logger.Info("Send req", "index", i)
			reqMgr.HandleRequest(&Request{id: strconv.Itoa(i), model: test.model})
		}()
	}

	wg.Wait()
	logger.Info("The end")
}
