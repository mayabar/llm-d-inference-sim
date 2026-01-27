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

package llmdinferencesim

import (
	"context"
	"errors"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	preprocessing "github.com/llm-d/llm-d-kv-cache-manager/pkg/preprocessing/chat_completions"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type Template interface {
	RenderChatTemplate(ctx context.Context, tempRequest *preprocessing.RenderJinjaTemplateRequest) (string, error)
}

type HFTemplate struct {
	// chatTemplateProcessor template processor,
	// used for creating a textual prompt for chat completions requests
	chatTemplateProcessor *preprocessing.ChatTemplatingProcessor
	// chatTemplate chat template
	chatTemplate string
	// chatTemplateKWArgs chat template parameters
	chatTemplateKWArgs map[string]interface{}
}

func CreateHFTemplate(ctx context.Context, config *common.Configuration) (*HFTemplate, error) {
	t := &HFTemplate{}

	t.chatTemplateProcessor = preprocessing.NewChatTemplatingProcessor()
	err := t.chatTemplateProcessor.Initialize()
	if err != nil {
		return nil, errors.Join(err, errors.New("failed to initialize chat templater"))
	}

	templateReq := preprocessing.FetchChatTemplateRequest{
		Model: config.Model,
		Token: config.HFToken,
	}
	t.chatTemplate, t.chatTemplateKWArgs, err = t.chatTemplateProcessor.FetchChatTemplate(ctx, templateReq)
	if err != nil {
		return nil, errors.Join(err, errors.New("failed to get chat template"))
	}

	logger := log.FromContext(ctx)

	logger.Info(">>> Template:\n", "", t.chatTemplate)
	return t, nil
}

func (hft *HFTemplate) RenderChatTemplate(ctx context.Context, tempRequest *preprocessing.RenderJinjaTemplateRequest) (string, error) {
	if tempRequest == nil {
		return "", errors.New("RenderChatTemplate called with nil render request")
	}
	logger := log.FromContext(ctx)

	tempRequest.ChatTemplate = hft.chatTemplate
	tempRequest.ChatTemplateKWArgs = hft.chatTemplateKWArgs

	response, err := hft.chatTemplateProcessor.RenderChatTemplate(ctx, tempRequest)
	if err != nil {
		logger.Error(err, "template rendering failed")
		return "", err
	}

	if len(response.RenderedChats) == 0 {
		return "", errors.New("no rendered chats found in response")
	}

	return response.RenderedChats[0], nil
}

type SimulationTemplate struct {
}

func CreateSimulationTemplate() (*SimulationTemplate, error) {
	return &SimulationTemplate{}, nil
}

func (st *SimulationTemplate) RenderChatTemplate(_ context.Context, _ *preprocessing.RenderJinjaTemplateRequest) (string, error) {
	return "", nil
}
