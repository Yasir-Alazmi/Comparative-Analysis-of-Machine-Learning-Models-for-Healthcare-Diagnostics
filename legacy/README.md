# Legacy scripts (not part of the results)

This folder keeps the early single-file benchmark scripts and the figures they produced, for reference only.

- `benchmarks/*.py`: standalone runs on the small public datasets (UCI heart failure, breast cancer, Pima diabetes, kidney disease, Kaggle stroke).
- `results/<dataset>/`: the bar, heatmap, radar and ROC figures and `metrics.csv` those scripts wrote.

**These scripts leak test information into training.** Several of them fill missing values with medians or modes computed on the full dataset (kidney disease, Pima diabetes, stroke), and they fit label encoders on the full dataset, all *before* `train_test_split`. Their numbers may therefore be optimistic and are not comparable with the main results.

None of the numbers in the main [README](../README.md) come from this folder. The main results come from `scripts/reproduce_benchmarks.py`, which fits every preprocessing step inside a cross-validated pipeline on the training data only (`src/preprocessor.py`). The same small datasets can be run through that leakage-free pipeline with `python run_benchmark.py --dataset <name>`.

To run a legacy script anyway, from the repository root:

```bash
python legacy/benchmarks/heart_failure_benchmark.py
```

It reads from `datasets/` and writes into `legacy/results/<dataset>/`.
