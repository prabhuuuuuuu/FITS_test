# Real_FITS: Multi-file ECG Training

## Prerequisites
- **Python**: 3.8+
- **Dependencies**:
  ```bash
  pip install -r requirements.txt
  ```
  Or install individually:
  ```bash
  pip install numpy pandas torch scikit-learn matplotlib scipy pyedflib wfdb
  ```

## Data Layout
Place EDF and QRS files in the `data` directory:
- `data/r01.edf`, `data/r01.qrs`
- `data/r04.edf`, `data/r04.qrs`
- `data/r07.edf`, `data/r07.qrs`
- `data/r08.edf`, `data/r08.qrs`
- `data/r10.edf`, `data/r10.qrs`

## Basic Training
Train with default settings (100 epochs, random file-level train/test split):
```bash
python multi_file_training.py --file_indices 1 4 7 8 10 --data_dir data
```

## Key Metrics
The script outputs only essential metrics: **Precision**, **Recall**, **F1**, **Accuracy**, and thresholds. Adjust thresholds at evaluation:
- `--peak_percentile 68`
- `--min_rr_sec 0.19`

**Example**:
```bash
python multi_file_training.py --file_indices 1 4 7 8 10 --data_dir data --peak_percentile 68 --min_rr_sec 0.19
```

## Important Arguments
| Argument              | Description                                  | Default          |
|-----------------------|----------------------------------------------|------------------|
| `--file_indices`      | Record indices (e.g., 1 4 7 8 10)           | -                |
| `--data_dir`          | Path to data folder                         | `data`           |
| `--epochs`            | Training epochs                             | `100`            |
| `--batch_size`        | Batch size                                  | `32`             |
| `--seq_len`           | Window length                               | `1024`           |
| `--stride`            | Window stride                               | `256`            |
| `--in_dataset_augmentation` | Enable noise augmentation             | -                |
| `--aug_rate`          | Noise std factor                            | `0.1`            |
| `--peak_percentile`   | Peak detection threshold percentile         | `90.0`           |
| `--min_rr_sec`        | Minimum RR interval (seconds)               | `0.25`           |
| `--tolerance_ms`      | Matching tolerance (milliseconds)           | `40`             |
| `--save_model`        | Output model path                           | `best_model.pth` |

## Examples
1. **Train with augmentation for 150 epochs**:
   ```bash
   python multi_file_training.py --file_indices 1 4 7 8 10 --data_dir data --epochs 150 --in_dataset_augmentation --aug_rate 0.15
   ```

2. **Evaluate with higher recall**:
   ```bash
   python multi_file_training.py --file_indices 1 4 7 8 10 --data_dir data --peak_percentile 65 --min_rr_sec 0.18
   ```

3. **Faster training preview**:
   ```bash
   python multi_file_training.py --file_indices 10 --data_dir data --epochs 20
   ```

## Notes
- **File-level split**: Prevents data leakage; use `--test_ratio` to adjust test size.
- **QRS annotations**: Read via WFDB from `rXX.qrs`. If missing/unreadable, synthetic peaks are generated as fallback.
