.DEFAULT_GOAL := help
SHELL := /bin/sh

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

up:  ## Start Postgres + Redis
	docker compose up -d --wait

down:  ## Stop them (keeps data)
	docker compose down

clean:  ## Stop and DELETE all indexed data
	docker compose down -v

install:  ## Create .venv and install everything
	uv sync --all-groups

ingest:  ## Ingest ./data/raw into the index
	uv run ragforge ingest data/raw

ask:  ## Ask one question:  make ask Q="what is X?"
	uv run ragforge ask "$(Q)"

api:  ## Run the API on :8000
	uv run uvicorn ragforge.api.main:app --reload --port 8000

ui:  ## Run the Streamlit UI on :8501
	uv run streamlit run ui/app.py

eval:  ## Run the eval harness and write the results table
	uv run ragforge eval

test:  ## Run tests
	uv run pytest -q

lint:  ## Lint and type-check
	uv run ruff check src tests
	uv run ruff format --check src tests

fmt:  ## Auto-format
	uv run ruff format src tests
	uv run ruff check --fix src tests

.PHONY: help up down clean install ingest ask api ui eval test lint fmt
