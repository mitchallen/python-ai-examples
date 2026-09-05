# python-ai-examples -- uv workspace of standalone AI examples.
#
# Every example lives in packages/<name> and is a workspace member, so one
# `uv sync` builds a single root .venv that can run any of them.

PKG ?= ai-python-101
UV  ?= uv

.DEFAULT_GOAL := help
.PHONY: help install sync test test-pkg run lint clean distclean

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

run: install ## Run one example (PKG=<name>); needs OPENAI_API_KEY
	$(UV) run --package $(PKG) $(PKG)

lint: install ## Byte-compile every example as a cheap syntax check
	$(UV) run python -m compileall -q packages

clean: ## Remove caches and build artifacts
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache packages/*/dist packages/*/build packages/*/*.egg-info

distclean: clean ## Also remove the virtualenv
	rm -rf .venv
