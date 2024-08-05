PYTHON := python3
PIP := pip3

install:
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) pipeline/pipeline.py pipeline/experiments/baseline.yaml

run-dev:
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) pipeline/pipeline.py pipeline/experiments/baseline.yaml