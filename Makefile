# Copyright 2025 The llm-d-inference-sim Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

LOCALBIN ?= $(shell pwd)/bin
$(LOCALBIN):
	mkdir -p $(LOCALBIN)

# Project-specific Go tooling is defined in Makefile.tools.mk.
include Makefile.tools.mk

# Makefile for the llm-d-inference-sim project

SHELL := /usr/bin/env bash

# Defaults
TARGETOS ?= $(shell go env GOOS)
TARGETARCH ?= $(shell go env GOARCH)
PROJECT_NAME ?= llm-d-inference-sim
IMAGE_REGISTRY ?= ghcr.io/llm-d
IMAGE_TAG_BASE ?= $(IMAGE_REGISTRY)/$(PROJECT_NAME)
SIM_TAG ?= dev
IMG = $(IMAGE_TAG_BASE):$(SIM_TAG)
POD_IP ?= pod
export POD_IP

ifeq ($(TARGETOS),darwin)
ifeq ($(TARGETARCH),amd64)
TOKENIZER_ARCH = x86_64
else
TOKENIZER_ARCH = $(TARGETARCH)
endif
else
TOKENIZER_ARCH = $(TARGETARCH)
endif

CONTAINER_TOOL := $(shell { command -v docker >/dev/null 2>&1 && echo docker; } || { command -v podman >/dev/null 2>&1 && echo podman; } || echo "")
BUILDER := $(shell command -v buildah >/dev/null 2>&1 && echo buildah || echo $(CONTAINER_TOOL))
PLATFORMS ?= linux/amd64 # linux/arm64 # linux/s390x,linux/ppc64le

# go source files
SRC = $(shell find . -type f -name '*.go')

.PHONY: help
help: ## Print help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

export PKG_CONFIG_PATH=/usr/lib/pkgconfig

##@ Python Configuration

PYTHON_VERSION := 3.12

# Unified Python configuration detection. This block runs once.
# It prioritizes python-config, then pkg-config, for reliability.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # macOS: Find Homebrew's python-config script for the most reliable flags.
    BREW_PREFIX := $(shell command -v brew >/dev/null 2>&1 && brew --prefix python@$(PYTHON_VERSION) 2>/dev/null)
    PYTHON_CONFIG := $(BREW_PREFIX)/bin/python$(PYTHON_VERSION)-config
    ifneq ($(shell $(PYTHON_CONFIG) --cflags 2>/dev/null),)
        PYTHON_CFLAGS := $(shell $(PYTHON_CONFIG) --cflags)
        # Use --ldflags --embed to get all necessary flags for linking
        PYTHON_LDFLAGS := $(shell $(PYTHON_CONFIG) --ldflags --embed)
        PYTHON_LIBS :=
    else
        $(error "Could not execute 'python$(PYTHON_VERSION)-config' from Homebrew. Please ensure Python is installed correctly with: 'brew install python@$(PYTHON_VERSION)'")
    endif
else ifeq ($(UNAME_S),Linux)
    # Linux: Use standard system tools to find flags.
    PYTHON_CONFIG := $(shell command -v python$(PYTHON_VERSION)-config || command -v python3-config)
    ifneq ($(shell $(PYTHON_CONFIG) --cflags 2>/dev/null),)
		# Use python-config if available and correct
        PYTHON_CFLAGS := $(shell $(PYTHON_CONFIG) --cflags)
        PYTHON_LDFLAGS := $(shell $(PYTHON_CONFIG) --ldflags --embed)
        PYTHON_LIBS :=
    else ifneq ($(shell pkg-config --cflags python-$(PYTHON_VERSION) 2>/dev/null),)
        # Fallback to pkg-config
        PYTHON_CFLAGS := $(shell pkg-config --cflags python-$(PYTHON_VERSION))
        PYTHON_LDFLAGS := $(shell pkg-config --libs python-$(PYTHON_VERSION))
        PYTHON_LIBS :=
    else
        $(error "Python $(PYTHON_VERSION) development headers not found. Please install with: 'sudo apt install python$(PYTHON_VERSION)-dev' or 'sudo dnf install python$(PYTHON_VERSION)-devel'")
    endif
else
    $(error "Unsupported OS: $(UNAME_S)")
endif

# Final CGO flags with all dependencies
CGO_CFLAGS_FINAL := $(PYTHON_CFLAGS) -Ilib
CGO_LDFLAGS_FINAL := $(PYTHON_LDFLAGS) $(PYTHON_LIBS) -Llib -ltokenizers -ldl -lm

VENV_DIR    ?= $(shell pwd)/.venv
VENV_BIN    := $(VENV_DIR)/bin
VENV_SRC  	:= $(VENV_DIR)/python

PYTHON_EXE := $(shell command -v python$(PYTHON_VERSION) || command -v python3)

GOMODCACHE := $(shell go env GOMODCACHE)
KV_CACHE_MGR_VERSION := $(shell go list -m -f '{{.Version}}' github.com/llm-d/llm-d-kv-cache-manager)
KV_CACHE_MGR_PATH := $(GOMODCACHE)/github.com/llm-d/llm-d-kv-cache-manager@$(KV_CACHE_MGR_VERSION)/pkg/preprocessing/chat_completions

# Common environment variables for Go tests and builds
export CGO_ENABLED=1
export CGO_CFLAGS=$(CGO_CFLAGS_FINAL)
export CGO_LDFLAGS=$(CGO_LDFLAGS_FINAL)
export PYTHONPATH=$(VENV_SRC):$(VENV_DIR)/lib/python$(PYTHON_VERSION)/site-packages

GO_LDFLAGS := -extldflags '-L$(shell pwd)/lib $(LDFLAGS) $(CGO_LDFLAGS)'
TOKENIZER_LIB = lib/libtokenizers.a
# Extract TOKENIZER_VERSION from Dockerfile
TOKENIZER_VERSION := $(shell grep '^ARG TOKENIZER_VERSION=' Dockerfile | cut -d'=' -f2)

.PHONY: download-tokenizer
download-tokenizer: $(TOKENIZER_LIB)
$(TOKENIZER_LIB):
	## Download the HuggingFace tokenizer bindings.
	@echo "Downloading HuggingFace tokenizer bindings for version $(TOKENIZER_VERSION)..."
	mkdir -p lib
	if [ "$(TARGETOS)" = "darwin" ] && [ "$(TARGETARCH)" = "amd64" ]; then \
		curl -L https://github.com/daulet/tokenizers/releases/download/$(TOKENIZER_VERSION)/libtokenizers.$(TARGETOS)-x86_64.tar.gz | tar -xz -C lib; \
	else \
		curl -L https://github.com/daulet/tokenizers/releases/download/$(TOKENIZER_VERSION)/libtokenizers.$(TARGETOS)-$(TARGETARCH).tar.gz | tar -xz -C lib; \
	fi
	ranlib lib/*.a

##@ Development

.PHONY: clean
clean:
	go clean -testcache -cache
	rm -f $(TOKENIZER_LIB)
	rmdir lib

.PHONY: format
format: ## Format Go source files
	@printf "\033[33;1m==== Running gofmt ====\033[0m\n"
	@gofmt -l -w $(SRC)

.PHONY: test
test: download-tokenizer install-python-deps # download-zmq ## Run unit tests
	@printf "\033[33;1m==== Running unit tests ====\033[0m\n"
	if [ -n "$(GINKGO_FOCUS)" ] && [ -z "$(GINKGO_FOCUS_PKG)" ]; then \
		echo "Error: GINKGO_FOCUS is defined without GINKGO_FOCUS_PKG. Both required or neither."; \
		exit 1; \
	elif [ -n "$(GINKGO_FOCUS)$(GINKGO_FOCUS_PKG)" ]; then \
		echo "Running specific tests"; \
		go test -v $(GINKGO_FOCUS_PKG) $(if $(GINKGO_FOCUS),-ginkgo.focus="$(GINKGO_FOCUS)",); \
	else \
		echo "Running all tests"; \
		go test -v ./pkg/...; \
	fi 

.PHONY: post-deploy-test
post-deploy-test: ## Run post deployment tests
	echo Success!
	@echo "Post-deployment tests passed."
	
.PHONY: lint
lint: $(GOLANGCI_LINT) ## Run lint
	@printf "\033[33;1m==== Running linting ====\033[0m\n"
	CGO_CFLAGS="$(CGO_CFLAGS)" $(GOLANGCI_LINT) run

##@ Build

.PHONY: build
build: check-go download-tokenizer install-python-deps download-zmq
	@printf "\033[33;1m==== Building ====\033[0m\n"
	CGO_CFLAGS="$(CGO_CFLAGS)" go build -ldflags="$(GO_LDFLAGS)" -o $(LOCALBIN)/$(PROJECT_NAME) cmd/$(PROJECT_NAME)/main.go

.PHONY: run
run: install-python-deps # build ## Run the application locally
	@printf "\033[33;1m==== Running application ====\033[0m\n"
	. $(VENV_DIR)/bin/activate && ./bin/$(PROJECT_NAME) $(ARGS)

##@ Container Build/Push

.PHONY:	image-build
image-build: check-container-tool ## Build Docker image ## Build Docker image using $(CONTAINER_TOOL)
	@printf "\033[33;1m==== Building Docker image $(IMG) ====\033[0m\n"
	$(CONTAINER_TOOL) build \
		--platform linux/$(TARGETARCH) \
 		--build-arg TARGETOS=linux \
		--build-arg TARGETARCH=$(TARGETARCH)\
		-t $(IMG) .

.PHONY: image-push
image-push: check-container-tool ## Push Docker image $(IMG) to registry
	@printf "\033[33;1m==== Pushing Docker image $(IMG) ====\033[0m\n"
	$(CONTAINER_TOOL) push $(IMG)

.PHONY: image-build-and-push
image-build-and-push: image-build image-push ## Build and push Docker image $(IMG) to registry

##@ Install/Uninstall Targets

# Default install/uninstall (Docker)
install: install-docker ## Default install using Docker
	@echo "Default Docker install complete."

uninstall: uninstall-docker ## Default uninstall using Docker
	@echo "Default Docker uninstall complete."

### Docker Targets

.PHONY: install-docker
install-docker: check-container-tool ## Install app using $(CONTAINER_TOOL)
	@echo "Starting container with $(CONTAINER_TOOL)..."
	$(CONTAINER_TOOL) run -d --name $(PROJECT_NAME)-container $(IMG)
	@echo "$(CONTAINER_TOOL) installation complete."
	@echo "To use $(PROJECT_NAME), run:"
	@echo "alias $(PROJECT_NAME)='$(CONTAINER_TOOL) exec -it $(PROJECT_NAME)-container /app/$(PROJECT_NAME)'"

.PHONY: uninstall-docker
uninstall-docker: check-container-tool ## Uninstall app from $(CONTAINER_TOOL)
	@echo "Stopping and removing container in $(CONTAINER_TOOL)..."
	-$(CONTAINER_TOOL) stop $(PROJECT_NAME)-container && $(CONTAINER_TOOL) rm $(PROJECT_NAME)-container
	@echo "$(CONTAINER_TOOL) uninstallation complete. Remove alias if set: unalias $(PROJECT_NAME)"

### Helm Targets
.PHONY: install-helm
install-helm: check-helm ## Install app using Helm
	@echo "Installing chart with Helm..."
	helm upgrade --install $(PROJECT_NAME) helm/$(PROJECT_NAME) --namespace default
	@echo "Helm installation complete."

.PHONY: uninstall-helm
uninstall-helm: check-helm ## Uninstall app using Helm
	@echo "Uninstalling chart with Helm..."
	helm uninstall $(PROJECT_NAME) --namespace default
	@echo "Helm uninstallation complete."

.PHONY: env
env: ## Print environment variables
	@echo "IMAGE_TAG_BASE=$(IMAGE_TAG_BASE)"
	@echo "IMG=$(IMG)"
	@echo "CONTAINER_TOOL=$(CONTAINER_TOOL)"

##@ Tools
.PHONY: check-go
check-go:
	@command -v go >/dev/null 2>&1 || { \
	  echo "❌ Go is not installed. Install it from https://golang.org/dl/"; exit 1; }

.PHONY: check-container-tool
check-container-tool:
	@command -v $(CONTAINER_TOOL) >/dev/null 2>&1 || { \
	  echo "❌ $(CONTAINER_TOOL) is not installed."; \
	  echo "🔧 Try: sudo apt install $(CONTAINER_TOOL) OR brew install $(CONTAINER_TOOL)"; exit 1; }

.PHONY: check-helm
check-helm:
	@command -v helm >/dev/null 2>&1 || { \
	  echo "❌ helm is not installed. Install it from https://helm.sh/docs/intro/install/"; exit 1; }

.PHONY: check-builder
check-builder:
	@if [ -z "$(BUILDER)" ]; then \
		echo "❌ No container builder tool (buildah, docker, or podman) found."; \
		exit 1; \
	else \
		echo "✅ Using builder: $(BUILDER)"; \
	fi

##@ Alias checking
.PHONY: check-alias
check-alias: check-container-tool
	@echo "🔍 Checking alias functionality for container '$(PROJECT_NAME)-container'..."
	@if ! $(CONTAINER_TOOL) exec $(PROJECT_NAME)-container /app/$(PROJECT_NAME) --help >/dev/null 2>&1; then \
	  echo "⚠️  The container '$(PROJECT_NAME)-container' is running, but the alias might not work."; \
	  echo "🔧 Try: $(CONTAINER_TOOL) exec -it $(PROJECT_NAME)-container /app/$(PROJECT_NAME)"; \
	else \
	  echo "✅ Alias is likely to work: alias $(PROJECT_NAME)='$(CONTAINER_TOOL) exec -it $(PROJECT_NAME)-container /app/$(PROJECT_NAME)'"; \
	fi

.PHONY: print-project-name
print-project-name: ## Print the current project name
	@echo "$(PROJECT_NAME)"

.PHONY: install-hooks
install-hooks: ## Install git hooks
	git config core.hooksPath hooks

.PHONY: detect-python
detect-python: ## Detects Python and prints the configuration.
	@printf "\033[33;1m==== Python Configuration ====\033[0m\n"
	@if [ -z "$(PYTHON_EXE)" ]; then \
		echo "ERROR: Python 3 not found in PATH."; \
		exit 1; \
	fi
	@# Verify the version of the found python executable using its exit code
	@if ! $(PYTHON_EXE) -c "import sys; sys.exit(0 if sys.version_info[:2] == ($(shell echo $(PYTHON_VERSION) | cut -d. -f1), $(shell echo $(PYTHON_VERSION) | cut -d. -f2)) else 1)"; then \
		echo "ERROR: Found Python at '$(PYTHON_EXE)' but it is not version $(PYTHON_VERSION)."; \
		echo "Please ensure 'python$(PYTHON_VERSION)' or a compatible 'python3' is in your PATH."; \
		exit 1; \
	fi
	@echo "Python executable: $(PYTHON_EXE) ($$($(PYTHON_EXE) --version))"
	@echo "Python CFLAGS:     $(PYTHON_CFLAGS)"
	@echo "Python LDFLAGS:    $(PYTHON_LDFLAGS)"
	@if [ -z "$(PYTHON_CFLAGS)" ]; then \
		echo "ERROR: Python development headers not found. See installation instructions above."; \
		exit 1; \
	fi
	@printf "\033[33;1m==============================\033[0m\n"

.PHONY: install-python-deps
install-python-deps: detect-python ## Sets up the Python virtual environment and installs dependencies.
	@printf "\033[33;1m==== Setting up Python virtual environment in $(VENV_DIR) ====\033[0m\n"
	@if [ ! -f "$(VENV_BIN)/pip" ]; then \
		echo "Creating virtual environment..."; \
		$(PYTHON_EXE) -m venv $(VENV_DIR) || { \
			echo "ERROR: Failed to create virtual environment."; \
			echo "Your Python installation may be missing the 'venv' module."; \
			echo "Try: 'sudo apt install python$(PYTHON_VERSION)-venv' or 'sudo dnf install python$(PYTHON_VERSION)-devel'"; \
			exit 1; \
		}; \
		mkdir -p $(VENV_SRC); \
	fi
	@echo "Upgrading pip and installing dependencies..."
	@$(VENV_BIN)/pip install --upgrade pip
	cp $(KV_CACHE_MGR_PATH)/requirements.txt $(VENV_SRC)/
	cp $(KV_CACHE_MGR_PATH)/render_jinja_template_wrapper.py $(VENV_SRC)/
	chmod u+w $(VENV_SRC)/*
	@$(VENV_BIN)/pip install -r $(VENV_SRC)/requirements.txt
	@echo "Verifying transformers installation..."
	@$(VENV_BIN)/python -c "import transformers; print('✅ Transformers version ' + transformers.__version__ + ' installed.')" || { \
		echo "ERROR: transformers library not properly installed in venv."; \
		exit 1; \
	}

##@ ZMQ Setup

.PHONY: download-zmq
download-zmq: ## Install ZMQ dependencies based on OS/ARCH
	@echo "Checking if ZMQ is already installed..."
	@if pkg-config --exists libzmq; then \
	  echo "✅ ZMQ is already installed."; \
	else \
	  echo "Installing ZMQ dependencies..."; \
	  if [ "$(TARGETOS)" = "linux" ]; then \
	    if [ -x "$$(command -v apt)" ]; then \
	      apt update && apt install -y libzmq3-dev; \
	    elif [ -x "$$(command -v dnf)" ]; then \
	      dnf install -y zeromq-devel; \
	    else \
	      echo "Unsupported Linux package manager. Install libzmq manually."; \
	      exit 1; \
	    fi; \
	  elif [ "$(TARGETOS)" = "darwin" ]; then \
	    if [ -x "$$(command -v brew)" ]; then \
	      brew install zeromq; \
	    else \
	      echo "Homebrew is not installed and is required to install zeromq. Install it from https://brew.sh/"; \
	      exit 1; \
	    fi; \
	  else \
	    echo "Unsupported OS: $(TARGETOS). Install libzmq manually - check https://zeromq.org/download/ for guidance."; \
	    exit 1; \
	  fi; \
	  echo "✅ ZMQ dependencies installed."; \
	fi
