/*
Copyright 2025 The llm-d-inference-sim Authors.

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
package kvcache

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/llm-d/llm-d-kv-cache-manager/pkg/kvcache/kvevents"
	"github.com/vmihailenco/msgpack/v5"
)

type EventAction int

const (
	eventActionStore EventAction = iota
	eventActionRemove
)

type EventData struct {
	action     EventAction
	hashValues []uint64
}

type Publisher struct{}

func (p *Publisher) PublishEvent(ctx context.Context, topic string, batch interface{}) error {
	// mock implementation
	fmt.Printf("Publish batch %#v\n", batch)
	return nil
}

type KVEventSender struct {
	mu           sync.RWMutex
	publisher    *Publisher
	topic        string
	eventChan    chan EventData
	maxBatchSize int
	delay        time.Duration
	batch        []msgpack.RawMessage
}

func NewKVEventSender(publisher *Publisher, topic string, ch chan EventData, maxBatchSize int, delay time.Duration) *KVEventSender {
	return &KVEventSender{
		publisher:    publisher,
		topic:        topic,
		eventChan:    ch,
		maxBatchSize: maxBatchSize,
		delay:        delay,
		batch:        make([]msgpack.RawMessage, 0, maxBatchSize),
	}
}

func (s *KVEventSender) Run(ctx context.Context) error {
	timer := time.NewTimer(s.delay)
	defer timer.Stop()

	for {
		select {
		case <-ctx.Done():
			// Before exit, publish remaining events if any
			err := s.publishHelper(ctx)
			if err != nil {
				return err
			}
			return ctx.Err()

		case eventData, ok := <-s.eventChan:
			if !ok {
				// Channel closed, publish remaining and exit
				return s.publishHelper(ctx)
			}

			// Encode eventData's hash value to msgpack.RawMessage
			var blockPayloadBytes msgpack.RawMessage
			var err error

			switch eventData.action {
			case eventActionStore:
				blockPayloadBytes, err = msgpack.Marshal(kvevents.BlockStored{BlockHashes: eventData.hashValues})
			case eventActionRemove:
				blockPayloadBytes, err = msgpack.Marshal(kvevents.BlockRemoved{BlockHashes: eventData.hashValues})
			default:
				return fmt.Errorf("invalid event action %d", eventData.action)
			}
			if err != nil {
				return fmt.Errorf("failed to marshal value: %w", err)
			}

			s.mu.Lock()
			s.batch = append(s.batch, blockPayloadBytes)
			defer s.mu.Unlock()

			// check if batch is big enough to be sent
			if len(s.batch) >= s.maxBatchSize {
				if err := s.publishHelper(ctx); err != nil {
					return err
				}

				// reset timer
				if !timer.Stop() {
					<-timer.C
				}
				timer.Reset(s.delay)
			}

		case <-timer.C:
			if err := s.publishHelper(ctx); err != nil {
				return err
			}
			timer.Reset(s.delay)
		}
	}
}

// helper to publish collected batch if not empty
func (s *KVEventSender) publishHelper(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.batch) == 0 {
		return nil
	}

	dpRank := 0
	eventBatch := kvevents.EventBatch{
		TS:               float64(time.Now().UnixNano()) / 1e9,
		Events:           s.batch,
		DataParallelRank: &dpRank,
	}

	err := s.publisher.PublishEvent(ctx, s.topic, eventBatch)

	// reset batch
	s.batch = make([]msgpack.RawMessage, 0, s.maxBatchSize)

	return err
}
