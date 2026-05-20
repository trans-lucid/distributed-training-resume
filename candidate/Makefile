.PHONY: dev logs seed test test-unit test-integration run clean

PYTHON ?= python3

dev:
	docker compose up -d

logs:
	docker compose logs -f

seed:
	$(PYTHON) -m src.run_store migrate
	$(PYTHON) -m src.checkpoint setup-storage

test:
	$(PYTHON) -m pytest tests/public

test-unit:
	$(PYTHON) -m pytest tests/public/test_checkpoint_unit.py

test-integration:
	$(PYTHON) -m pytest tests/public/test_integration_resume.py

run:
	$(PYTHON) -m src.report --out results/resume_report.json

clean:
	docker compose down -v
