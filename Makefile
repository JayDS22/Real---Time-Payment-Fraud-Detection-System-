# Fraud Detection System Makefile

.PHONY: help install setup train test run clean docker deploy monitor

# Variables
PYTHON := python3
PIP := pip3
DOCKER := docker
DOCKER_COMPOSE := docker-compose
KUBECTL := kubectl

# Default target
help: ## Show this help message
	@echo "Fraud Detection System - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Setup and Installation
install: ## Install Python dependencies
	$(PIP) install -r requirements.txt

setup: ## Setup development environment
	@echo "Setting up fraud detection system..."
	$(PYTHON) -m venv venv
	./venv/bin/pip install -r requirements.txt
	cp .env.example .env
	mkdir -p data models logs reports
	@echo "✅ Setup complete! Activate virtual environment with: source venv/bin/activate"

install-dev: ## Install development dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black flake8 mypy pre-commit
	pre-commit install

# Data and Model Training
generate-data: ## Generate synthetic training data
	$(PYTHON) src/training/train_model.py --generate-only

train: ## Train fraud detection models
	@echo "Training fraud detection models..."
	$(PYTHON) src/training/train_model.py
	@echo "✅ Model training complete!"

train-fast: ## Quick training with smaller dataset
	$(PYTHON) src/training/train_model.py --samples 10000 --trials 10

# Testing
test: ## Run unit tests
	$(PYTHON) -m pytest tests/ -v

test-coverage: ## Run tests with coverage
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=html --cov-report=term

test-api: ## Test API endpoints
	$(PYTHON) -m pytest tests/test_api.py -v

test-integration: ## Run integration tests
	$(PYTHON) -m pytest tests/integration/ -v

lint: ## Run code linting
	black src/ tests/ --check
	flake8 src/ tests/
	mypy src/

format: ## Format code
	black src/ tests/

# Services
start-infra: ## Start infrastructure services (Redis, Kafka, etc.)
	$(DOCKER_COMPOSE) up -d redis kafka zookeeper postgres mlflow prometheus grafana
	@echo "✅ Infrastructure services started"
	@echo "Redis: localhost:6379"
	@echo "Kafka: localhost:9092"
	@echo "PostgreSQL: localhost:5432"
	@echo "MLflow: http://localhost:5000"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000 (admin/admin123)"

stop-infra: ## Stop infrastructure services
	$(DOCKER_COMPOSE) down

restart-infra: ## Restart infrastructure services
	$(DOCKER_COMPOSE) restart

# Application Services
start-api: ## Start fraud detection API
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

start-consumer: ## Start Kafka consumer
	$(PYTHON) src/streaming/kafka_consumer.py

start-producer: ## Start transaction producer (simulation)
	$(PYTHON) src/streaming/transaction_producer.py --mode simulate --duration 60 --tps 10

start-feature-store: ## Start feature store service
	$(PYTHON) src/features/feature_store.py

start-dashboard: ## Start monitoring dashboard
	streamlit run src/monitoring/dashboard.py --server.port=8501 --server.address=0.0.0.0

# Full System
run-all: ## Start all services
	$(DOCKER_COMPOSE) up -d
	@echo "✅ All services started!"
	@echo "API: http://localhost:8000"
	@echo "Dashboard: http://localhost:8501"
	@echo "Grafana: http://localhost:3000"

stop-all: ## Stop all services
	$(DOCKER_COMPOSE) down

logs: ## Show logs from all services
	$(DOCKER_COMPOSE) logs -f

# Docker
build: ## Build Docker images
	$(DOCKER) build -f docker/Dockerfile.api -t fraud-detection:latest .
	$(DOCKER) build -f docker/Dockerfile.consumer -t fraud-consumer:latest .
	$(DOCKER) build -f docker/Dockerfile.monitoring -t fraud-monitoring:latest .

push: ## Push Docker images to registry
	$(DOCKER) tag fraud-detection:latest your-registry/fraud-detection:latest
	$(DOCKER) push your-registry/fraud-detection:latest

# Kubernetes
deploy-k8s: ## Deploy to Kubernetes
	$(KUBECTL) apply -f kubernetes/
	@echo "✅ Deployed to Kubernetes"
	@echo "Check status with: kubectl get pods -n fraud-detection"

delete-k8s: ## Delete Kubernetes deployment
	$(KUBECTL) delete -f kubernetes/

k8s-status: ## Show Kubernetes deployment status
	$(KUBECTL) get all -n fraud-detection

k8s-logs: ## Show Kubernetes logs
	$(KUBECTL) logs -f deployment/fraud-api -n fraud-detection

# Data Operations
load-sample-data: ## Load sample transaction data
	$(PYTHON) -c "
import pandas as pd
from src.streaming.transaction_producer import TransactionProducer
producer = TransactionProducer()
transactions = [producer.generate_transaction() for _ in range(1000)]
pd.DataFrame(transactions).to_csv('data/sample_transactions.csv', index=False)
print('Generated 1000 sample transactions')
"

simulate-traffic: ## Simulate realistic traffic for 10 minutes
	$(PYTHON) src/streaming/transaction_producer.py --mode simulate --duration 10 --tps 5

simulate-fraud-attack: ## Simulate fraud attack
	$(PYTHON) src/streaming/transaction_producer.py --mode attack --duration 5 --intensity 10

# Monitoring and Maintenance
monitor: ## Open monitoring dashboard
	@echo "Opening monitoring interfaces..."
	@echo "API Health: http://localhost:8000/health"
	@echo "Metrics: http://localhost:8000/metrics"
	@echo "Dashboard: http://localhost:8501"
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"

check-health: ## Check system health
	@echo "Checking system health..."
	@curl -s http://localhost:8000/health | jq '.' || echo "API not responding"
	@curl -s http://localhost:8501 > /dev/null && echo "✅ Dashboard: OK" || echo "❌ Dashboard: Failed"

performance-test: ## Run performance tests
	locust -f tests/load/locustfile.py --headless -u 50 -r 10 -t 60s --host=http://localhost:8000

# Maintenance
clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage

clean-docker: ## Clean up Docker containers and images
	$(DOCKER) system prune -f
	$(DOCKER) volume prune -f

reset-data: ## Reset all data (WARNING: destructive)
	$(DOCKER_COMPOSE) down -v
	rm -rf data/* logs/* models/* reports/*
	@echo "⚠️ All data has been reset!"

backup-models: ## Backup trained models
	@mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	@cp -r models/* backups/$(shell date +%Y%m%d_%H%M%S)/
	@echo "✅ Models backed up to backups/$(shell date +%Y%m%d_%H%M%S)/"

# Development
dev-setup: ## Setup development environment with hot reload
	@echo "Starting development environment..."
	$(DOCKER_COMPOSE) up -d redis kafka zookeeper postgres
	@echo "Infrastructure ready. Start development servers with:"
	@echo "  make start-api     (in terminal 1)"
	@echo "  make start-consumer (in terminal 2)"
	@echo "  make start-dashboard (in terminal 3)"

debug: ## Start services in debug mode
	$(PYTHON) -m debugpy --listen 5678 --wait-for-client src/api/main.py

jupyter: ## Start Jupyter notebook for data exploration
	jupyter notebook notebooks/

# CI/CD
ci-test: ## Run CI tests
	$(PYTHON) -m pytest tests/ --junitxml=test-results.xml --cov=src --cov-report=xml

ci-build: ## Build for CI
	$(DOCKER) build -f docker/Dockerfile.api -t fraud-detection:$(shell git rev-parse --short HEAD) .

ci-security: ## Run security checks
	bandit -r src/
	safety check

# Documentation
docs: ## Generate documentation
	@echo "Generating documentation..."
	@mkdir -p docs/api
	@echo "API documentation will be available at http://localhost:8000/docs when API is running"

# Database
db-migrate: ## Run database migrations
	alembic upgrade head

db-reset: ## Reset database
	alembic downgrade base
	alembic upgrade head

db-seed: ## Seed database with sample data
	$(PYTHON) scripts/seed_database.py

# Model Management
model-info: ## Show model information
	@echo "Model Information:"
	@ls -la models/ 2>/dev/null || echo "No models found. Run 'make train' first."

model-validate: ## Validate model performance
	$(PYTHON) src/training/validate_model.py

model-deploy: ## Deploy model to production
	@echo "Deploying model to production..."
	@echo "This would typically involve MLflow model registry operations"

# Utility commands
version: ## Show version information
	@echo "Fraud Detection System v1.0.0"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Docker: $(shell $(DOCKER) --version)"
	@echo "Kubernetes: $(shell $(KUBECTL) version --client --short 2>/dev/null || echo 'Not installed')"

env-check: ## Check environment setup
	@echo "Environment Check:"
	@$(PYTHON) --version || echo "❌ Python not found"
	@$(DOCKER) --version || echo "❌ Docker not found"
	@$(DOCKER_COMPOSE) --version || echo "❌ Docker Compose not found"
	@redis-cli --version || echo "⚠️ Redis CLI not found (optional)"
	@echo "✅ Environment check complete"

# Examples and demos
demo-basic: ## Run basic fraud detection demo
	@echo "Running basic fraud detection demo..."
	$(PYTHON) examples/basic_demo.py

demo-realtime: ## Run real-time demo
	@echo "Starting real-time fraud detection demo..."
	@echo "This will start producer, consumer, and dashboard"
	$(MAKE) start-infra
	sleep 10
	$(PYTHON) src/streaming/transaction_producer.py --mode simulate --duration 5 --tps 2 &
	$(PYTHON) src/streaming/kafka_consumer.py &
	streamlit run src/monitoring/dashboard.py

# Troubleshooting
troubleshoot: ## Run troubleshooting diagnostics
	@echo "=== Troubleshooting Fraud Detection System ==="
	@echo ""
	@echo "1. Checking Python environment..."
	@$(PYTHON) --version
	@echo ""
	@echo "2. Checking required packages..."
	@$(PIP) list | grep -E "(fastapi|kafka|redis|sklearn|xgboost|lightgbm)" || echo "Some packages missing"
	@echo ""
	@echo "3. Checking Docker services..."
	@$(DOCKER_COMPOSE) ps || echo "Docker services not running"
	@echo ""
	@echo "4. Checking ports..."
	@netstat -tuln | grep -E "(6379|9092|5432|8000|8501)" || echo "Some services may not be running"
	@echo ""
	@echo "5. Checking model files..."
	@ls -la models/ || echo "No model files found - run 'make train'"
	@echo ""
	@echo "=== End Troubleshooting ==="

# Quick commands for common workflows
quick-start: ## Quick start for development (train + run)
	@echo "🚀 Quick starting fraud detection system..."
	$(MAKE) setup
	$(MAKE) start-infra
	@echo "⏳ Waiting for infrastructure..."
	sleep 15
	$(MAKE) train-fast
	$(MAKE) start-api &
	sleep 5
	$(MAKE) start-dashboard &
	@echo ""
	@echo "✅ System is starting up!"
	@echo "📊 Dashboard: http://localhost:8501"
	@echo "🔌 API: http://localhost:8000"
	@echo "📈 Health: http://localhost:8000/health"

full-demo: ## Full system demonstration
	@echo "🎭 Starting full fraud detection demonstration..."
	$(MAKE) quick-start
	sleep 10
	@echo "🏭 Starting transaction simulation..."
	$(MAKE) simulate-traffic &
	@echo ""
	@echo "🎯 Demo is running!"
	@echo "View the dashboard at http://localhost:8501 to see real-time fraud detection"

# Production commands
prod-deploy: ## Deploy to production environment
	@echo "🚀 Deploying to production..."
	@echo "⚠️ Make sure you have configured production settings!"
	@read -p "Are you sure you want to deploy to production? (y/N) " confirm && [ "$confirm" = "y" ]
	$(MAKE) build
	$(MAKE) deploy-k8s
	@echo "✅ Production deployment complete"

prod-scale: ## Scale production deployment
	$(KUBECTL) scale deployment fraud-api --replicas=5 -n fraud-detection
	@echo "✅ Scaled fraud-api to 5 replicas"

prod-rollback: ## Rollback production deployment
	$(KUBECTL) rollout undo deployment/fraud-api -n fraud-detection
	@echo "✅ Rolled back fraud-api deployment"

# Aliases for convenience
up: run-all ## Alias for run-all
down: stop-all ## Alias for stop-all
build-all: build ## Alias for build
restart: stop-all run-all ## Restart all services

# Default goal
.DEFAULT_GOAL := help
