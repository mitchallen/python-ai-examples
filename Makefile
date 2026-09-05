# python-ai-examples -- uv workspace of standalone AI examples.
#
# Every example lives in packages/<name> and is a workspace member, so one
# `uv sync` builds a single root .venv that can run any of them.

PKG  ?= ai-python-101
ARGS ?=
UV   ?= uv

# Local Ollama, for running the examples with no API key and no cost.
OLLAMA_MODEL    ?= llama3.2:3b
OLLAMA_BASE_URL ?= http://localhost:11434/v1

.DEFAULT_GOAL := help
.PHONY: help install sync test test-pkg run run-ollama lint badge badge-check clean distclean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  Override the target package with PKG=<name>, e.g. make run PKG=ai-python-101"

install sync: ## Create/refresh the workspace .venv with every member installed
	$(UV) sync --all-packages

test: install ## Run the whole workspace test suite
	$(UV) run pytest

test-pkg: install ## Run just one package's tests (PKG=<name>)
	$(UV) run pytest packages/$(PKG)

run: install ## Run one example (PKG=<name>, ARGS=...); needs OPENAI_API_KEY
	$(UV) run --package $(PKG) $(PKG) $(ARGS)

run-ollama: install ## Run one example against local Ollama (no API key, no cost)
	@curl -fsS -m 3 $(OLLAMA_BASE_URL:/v1=)/api/version >/dev/null 2>&1 || { \
		echo "No Ollama at $(OLLAMA_BASE_URL) -- start it with: ollama serve"; exit 1; }
	@ollama list 2>/dev/null | awk 'NR>1 {print $$1}' | grep -qx "$(OLLAMA_MODEL)" || { \
		echo "Model $(OLLAMA_MODEL) is not pulled -- get it with: ollama pull $(OLLAMA_MODEL)"; exit 1; }
	OPENAI_API_KEY=ollama \
	OPENAI_BASE_URL=$(OLLAMA_BASE_URL) \
	OPENAI_MODEL=$(OLLAMA_MODEL) \
	$(UV) run --package $(PKG) $(PKG) $(ARGS)

badge: install ## Regenerate the README test-count badge from pytest
	$(UV) run python scripts/test_badge.py

badge-check: install ## Fail if the committed test-count badge is stale (CI runs this)
	$(UV) run python scripts/test_badge.py --check

lint: install ## Byte-compile every example as a cheap syntax check
	$(UV) run python -m compileall -q packages

clean: ## Remove caches and build artifacts
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache packages/*/dist packages/*/build packages/*/*.egg-info

distclean: clean ## Also remove the virtualenv
	rm -rf .venv
