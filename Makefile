# Makefile

# Variables
VENV_NAME = .venv
PYTHON = $(VENV_NAME)/bin/python
PIP = $(VENV_NAME)/bin/pip
ACTIVATE = source $(VENV_NAME)/bin/activate

# Targets

.PHONY: venv install preprocess train clean lint serve all

venv:
	python3 -m venv $(VENV_NAME)
	@echo "✅ Virtual environment created in $(VENV_NAME)"

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed"

preprocess:
	$(PYTHON) src/data_preprocessing.py all

train:
	$(PYTHON) src/train.py

serve:
	$(PYTHON) src/app.py

lint:
	black src/ --check

clean:
	rm -rf __pycache__ .pytest_cache data/processed/*.csv
	rm -rf $(VENV_NAME)
	@echo "🧹 Project cleaned"

all: install preprocess train
	@echo "🚀 All steps complete!"
