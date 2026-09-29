.PHONY: install train simulate serve serve-docker test test-cov lint clean

# ── Environment ────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── Pipeline ───────────────────────────────────────────────────────────────
train:
	python scripts/train.py --n-samples 200000 --fraud-rate 0.005

simulate:
	python scripts/simulate_stream.py --n-transactions 5000 --inject-drift

# ── Serving ────────────────────────────────────────────────────────────────
serve:
	uvicorn src.api.main:app --reload --port 8000

serve-docker:
	docker compose up --build

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# ── Testing ────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --asyncio-mode=auto

test-cov:
	pytest tests/ -v --asyncio-mode=auto --cov=src --cov-report=term-missing

# ── Code quality ───────────────────────────────────────────────────────────
lint:
	ruff check src/ scripts/ tests/
	mypy src/ --ignore-missing-imports

# ── Cleanup ────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
