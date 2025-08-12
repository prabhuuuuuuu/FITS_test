README — Multi-file ECG Training (Real_FITS)
Prerequisites

Python 3.8+
Install dependencies:pip install -r requirements.txt

Or individually:pip install numpy pandas torch scikit-learn matplotlib scipy pyedflib wfdb



Data Layout

Place EDF and QRS files in the data directory:data/r01.edf, data/r01.qrs
data/r04.edf, data/r04.qrs
data/r07.edf, data/r07.qrs
data/r08.edf, data/r08.qrs
data/r10.edf, data/r10.qrs



Basic Training

Train with default settings (100 epochs), random file-level train/test split:python multi_file_training.py --file_indices 1 4 7 8 10 --data_dir data



Minimal Output with Key Metrics

The script prints only key metrics (Precision, Recall, F1, Accuracy) and thresholds:
Adjust thresholds at evaluation time:
--peak_percentile 68
--min_rr_sec 0.19


Example:python multi_file_training.py --file_indices 1 4 7 8 10 --data_dir data --peak_percentile 68 --min_rr_sec 0.19





Important Arguments

--file_indices: List of record indices (e.g., 1 4 7 8 10)
--data_dir: Path to data folder (default: data)
--epochs: Training epochs (default: 100)
--batch_size: Batch size (default: 32)
--seq_len: Window length (default: 1024)
--stride: Window stride (default: 256)
--in_dataset_augmentation: Enable noise augmentation
--aug_rate: Noise std factor (default: 0.1)
--peak_percentile: Peak detection threshold percentile (default: 90.0)
--min_rr_sec: Minimum RR interval in seconds (default: 0.25)
--tolerance_ms: Matching tolerance in milliseconds (default: 40)
--save_model: Output model path (default: best_model.pth)

Examples

Train with augmentation for 150 epochs:
python multi_file_training.py --file_indices 1 4 7 8 10 --data_dir data --epochs 150 --in_dataset_augmentation --aug_rate 0.15


Evaluate with more sensitive detection (higher recall):
python multi_file_training.py --file_indices 1 4 7 8 10 --data_dir data --peak_percentile 65 --min_rr_sec 0.18


Faster training preview:
python multi_file_training.py --file_indices 10 --data_dir data --epochs 20



Notes

The split is at file level to avoid leakage; set --test_ratio to control test size.
QRS annotations are read via WFDB from rXX.qrs; if missing/unreadable, synthetic peaks are generated as fallback.
