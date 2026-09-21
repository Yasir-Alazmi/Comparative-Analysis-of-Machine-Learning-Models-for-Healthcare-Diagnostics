.PHONY: help data benchmark smoke reproduce test clean

PYTHON ?= python

help:
	@echo "Trustworthy Clinical AI Benchmark Suite - Command Reference"
	@echo "------------------------------------------------------------"
	@echo "make data        - Download and assemble authentic CDC NHANES and Framingham datasets"
	@echo "make benchmark   - Run full clinical benchmark with 5-fold CV on NHANES cohort"
	@echo "make smoke       - Fast smoke benchmark for quick sanity validation"
	@echo "make reproduce   - Single-command full deterministic scientific reproduction"
	@echo "make test        - Run complete pytest integrity suite"
	@echo "make clean       - Remove cache files and temporary build artifacts"

data:
	$(PYTHON) scripts/build_real_datasets.py

benchmark:
	$(PYTHON) run_benchmark.py --dataset nhanes_cardiovascular --cv 5 --seed 42

smoke:
	$(PYTHON) run_benchmark.py --dataset nhanes_cardiovascular --fast --seed 42

reproduce:
	$(PYTHON) scripts/reproduce_benchmarks.py

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
