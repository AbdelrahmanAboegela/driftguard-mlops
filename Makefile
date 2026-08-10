.PHONY: help install data train serve simulate test docker-up docker-down clean

PYTHON ?= python

help:
	@echo "DriftGuard MLOps Management Commands:"
	@echo "  make install       - Install project dependencies"
	@echo "  make data          - Download and temporally split credit card fraud dataset"
	@echo "  make train         - Train baseline Champion model and register to MLflow"
	@echo "  make serve         - Start FastAPI inference service on port 8000"
	@echo "  make simulate      - Run full end-to-end drift and auto-retraining simulation"
	@echo "  make test          - Run test suite with pytest"
	@echo "  make docker-up     - Launch FastAPI, MLflow, Prometheus, and Grafana in Docker"
	@echo "  make docker-down   - Stop all Docker containers"

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) data/split_data.py

train:
	$(PYTHON) training/train.py

serve:
	uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload

simulate:
	$(PYTHON) scripts/simulate_production.py

test:
	pytest tests/ -v

docker-up:
	docker-compose up -d --build

docker-down:
	docker-compose down

clean:
	rm -rf data/processed/* data/raw/* reports/* mlflow.db .pytest_cache
