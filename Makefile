.PHONY: test lint run venv

# journal-api-ai (FastAPI AI worker) — project mandiri.
venv:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

test:
	.venv/bin/python -m pytest -q

lint:
	.venv/bin/python -m pytest -q

run:
	.venv/bin/uvicorn app.main:app --port 8000 --host 127.0.0.1