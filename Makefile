install:
	pip install -r requirements.txt

train:
	python -m src.train.train

test:
	pytest tests/

serve:
	uvicorn src.serve.app:app --reload

drift:
	python -m src.monitor.drift
