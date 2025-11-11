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

package dataset

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
)

type CustomDataset struct {
	BaseDataset
	sqliteHelper *sqliteHelper
}

const (
	progressLogTimeInterval    = 5 * time.Second
	progressLogPercentInterval = 10
)

func (d *CustomDataset) downloadDataset(ctx context.Context, url string, path string) error {
	folder := filepath.Dir(path)
	err := os.MkdirAll(folder, 0755)
	if err != nil {
		return fmt.Errorf("failed to create parent directory: %w", err)
	}

	if _, err := os.Stat(path); err == nil {
		// file already exists
		return errors.New("Dataset file already exists, should not download: " + path)
	}

	out, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() {
		cerr := out.Close()
		if cerr != nil {
			d.logger.Error(cerr, "failed to close file after download")
		}
	}()

	d.logger.V(logging.INFO).Info("Using dataset-url", "dataset-url", url)
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer func() {
		cerr := resp.Body.Close()
		if cerr != nil {
			d.logger.Error(cerr, "failed to close response body after download")
		}
	}()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	// Progress reader with context
	pr := &progressReader{
		Reader:    resp.Body,
		total:     resp.ContentLength,
		logger:    d.logger,
		ctx:       ctx,
		startTime: time.Now(),
	}

	written, err := io.Copy(out, pr)
	if err != nil {
		// Remove incomplete file
		cerr := os.Remove(path)
		if cerr != nil {
			d.logger.Error(cerr, "failed to remove incomplete file after download")
		}
		// If context was cancelled, return a specific error
		if errors.Is(err, context.Canceled) {
			return errors.New("download cancelled by user")
		}
		return fmt.Errorf("failed to download file: %w", err)
	}
	// Check if file size is zero
	if written == 0 {
		cerr := os.Remove(path)
		if cerr != nil {
			d.logger.Error(cerr, "failed to remove empty file after download")
		}
		return errors.New("downloaded file is empty")
	}

	// Ensure file is fully flushed and closed before returning success
	if err := out.Sync(); err != nil {
		cerr := os.Remove(path)
		if cerr != nil {
			d.logger.Error(cerr, "failed to remove incomplete file after download")
		}
		return fmt.Errorf("failed to sync file: %w", err)
	}

	return nil
}

// progressReader wraps an io.Reader and logs download progress.
type progressReader struct {
	io.Reader
	total       int64
	downloaded  int64
	startTime   time.Time
	lastPct     int
	lastLogTime time.Time
	logger      logr.Logger
	ctx         context.Context
}

func (pr *progressReader) Read(p []byte) (int, error) {
	select {
	case <-pr.ctx.Done():
		return 0, pr.ctx.Err()
	default:
	}
	n, err := pr.Reader.Read(p)
	pr.downloaded += int64(n)
	if pr.total > 0 {
		pct := int(float64(pr.downloaded) * 100 / float64(pr.total))
		now := time.Now()

		timeSinceLastLog := now.Sub(pr.lastLogTime).Seconds()
		pctDiff := pct - pr.lastPct

		if timeSinceLastLog >= progressLogTimeInterval.Seconds() || (pctDiff >= progressLogPercentInterval && pct != pr.lastPct) {
			// progress will be shown every interval seconds or every interval percent of progress
			pr.logProgress(pct)
			pr.lastPct = pct
			pr.lastLogTime = now
		}
	}
	return n, err
}

func (pr *progressReader) logProgress(pct int) {
	elapsedTime := time.Since(pr.startTime).Seconds()
	speed := float64(pr.downloaded) / (1024 * 1024 * elapsedTime)
	remainingTime := float64(pr.total-pr.downloaded) / (float64(pr.downloaded) / elapsedTime)
	if pct != 100 {
		pr.logger.V(logging.INFO).Info("Dataset download progress", "%", pct, "speed (MB/s)", speed, "remaining time (s)", remainingTime)
	} else {
		pr.logger.V(logging.INFO).Info("Download completed", "average speed (MB/s)", speed, "total time (s)", elapsedTime)
	}
}

func (d *CustomDataset) Init(ctx context.Context, logger logr.Logger, random *common.Random,
	path string, url string, useInMemory bool, maxModelLen int) error {
	if err := d.BaseDataset.Init(ctx, logger, random, path, url, useInMemory, maxModelLen); err != nil {
		return err
	}

	d.sqliteHelper = newSqliteHelper(logger)

	if path == "" {
		return errors.New("no dataset path provided")
	}
	if url == "" {
		d.logger.V(logging.INFO).Info("Using dataset from", "path", path)
		return d.sqliteHelper.connectToDB(path, useInMemory)
	}
	_, err := os.Stat(path)
	if err != nil {
		// file does not exist, download it
		err = d.downloadDataset(ctx, url, path)
		if err != nil {
			// if the file is created but incomplete, remove it
			if _, statErr := os.Stat(path); statErr == nil {
				cerr := os.Remove(path)
				if cerr != nil {
					d.logger.Error(cerr, "failed to remove incomplete file after download")
				}
			}
			return fmt.Errorf("failed to download dataset: %w", err)
		}
	}
	d.logger.V(logging.INFO).Info("Using dataset path", "dataset-path", path)

	return d.sqliteHelper.connectToDB(path, useInMemory)
}

func (d *CustomDataset) getPromptHash(req openaiserverapi.CompletionRequest) []byte {
	hashArray := sha256.Sum256([]byte(req.GetFullPrompt()))
	return hashArray[:]
}

func (d *CustomDataset) getPromptHashHex(hashBytes []byte) string {
	return hex.EncodeToString(hashBytes)
}

// categorizeResponses receives list of responses tokens and maximum response length
// categorize responses to three collections:
// - shorter or equal length to maxLen
// - exact maxLen length
// - longer than maxLen
func (d *CustomDataset) categorizeResponses(responses [][]string, maxLen int) (shorterOrEqLen [][]string, equalLen [][]string, longerLen [][]string) {
	for _, respTokens := range responses {
		switch {
		case len(respTokens) == maxLen:
			shorterOrEqLen = append(shorterOrEqLen, respTokens)
			equalLen = append(equalLen, respTokens)
		case len(respTokens) < maxLen:
			shorterOrEqLen = append(shorterOrEqLen, respTokens)
		default:
			longerLen = append(longerLen, respTokens)
		}
	}
	return
}

// getRandomResponse returns a randomly selected element from the given array, array is not empty
func (d *CustomDataset) getRandomResponse(responses [][]string) []string {
	return responses[d.random.RandomInt(0, len(responses)-1)]
}

// GetTokens returns tokens and finishReason for the given request and mode (echo or random)
// In echo mode the prompt is returned.
// In random mode follow this steps:
// Calculate maximum length of response (basedon max-tokens or max-completions-tokens or model-len)
// If dataset contains responses for the given prompt, and there are responses with length <=
// max response length - use random one from the list,
// otherwise select random one from the longer responses and trim it as required
// If no responses were found in the dataset for the given prompt,
// get random record fromn the dataset with response length equal or lower than max response length,
// if there is no records shorter/equal to max length - get random response from the dataset
// and trim it to the required length
// if ignore_eos=true the response always will have the max response len tokens, missing tokens
// are randomly selected from the hard-coded collection
func (d *CustomDataset) GetTokens(req openaiserverapi.CompletionRequest, mode string) ([]string, string, error) {
	if mode == common.ModeEcho {
		return d.getTokensInEchoMode(req)
	}

	maxResponseLen, _ := d.calculateResponseMaxLen(req)
	responseTokens := []string{}

	// get all records for the hashes prompt
	promptHash := d.getPromptHash(req)
	promptHashHex := d.getPromptHashHex(promptHash)
	responses, err := d.sqliteHelper.getResponsesForPrompt(promptHashHex)
	if err != nil {
		return responseTokens, "", err
	}

	if len(responses) > 0 {
		// has responses for the given request
		shorterOrEqLenResponses, equalLenResponses, longerLenResponses := d.categorizeResponses(responses, maxResponseLen)

		if req.GetIgnoreEOS() {
			// must return response with exactly calculated max length
			switch {
			case len(equalLenResponses) > 0:
				// has responses with required length - return randomly selected response
				responseTokens = d.getRandomResponse(equalLenResponses)
			case len(longerLenResponses) > 0:
				// has responses longer than required - return randomly selected trimmed response
				responseTokens = d.getRandomResponse(longerLenResponses)[:maxResponseLen]
			default:
				// all responses are shorter than required, select randomly and pad with random tokens
				responseTokens = d.getRandomResponse(shorterOrEqLenResponses)
				responseTokens = append(responseTokens, d.generatePresetRandomTokens(maxResponseLen-len(responseTokens))...)
			}
		} else {
			// has responses for the request, return response shorter or equal to the maxReponsesLen
			// finishReason = common.LengthFinishReason
			if len(shorterOrEqLenResponses) > 0 {
				// has responses shorter or equal length than required - return randomly selected response
				responseTokens = d.getRandomResponse(shorterOrEqLenResponses)
			} else {
				// all responses are longer than required, use randomly sleected trimmed response
				responseTokens = d.getRandomResponse(longerLenResponses)[:maxResponseLen]
			}
		}
	} else {
		// no resopnses for the given request
		// try to find a random response with number of tokens <= tokens limit
		randomResponses, err := d.sqliteHelper.getResponsesForLen(maxResponseLen)
		if err != nil {
			return responseTokens, "", err
		}
		if len(randomResponses) == 0 {
			// failed to get response with number of tokens <= tokensLimit, get response with any number of tokens
			randomResponses, err = d.sqliteHelper.getRandomResponse()
			if err != nil {
				return responseTokens, "", err
			}
			if len(randomResponses) == 0 {
				// shouldn't happen
				return responseTokens, "", errors.New("Dataset is empty")
			}
		}
		// response has too much tokens, trim it
		responseTokens = randomResponses[0][:maxResponseLen]
	}

	finishReason := common.StopFinishReason
	if len(responseTokens) == maxResponseLen {
		finishReason = common.LengthFinishReason
	}

	return responseTokens, finishReason, nil
}
