import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

try:
    import pyedflib
except ImportError:
    raise ImportError("Please install pyedflib: pip install pyedflib")
try:
    from scipy.signal import find_peaks
except ImportError:
    raise ImportError("Please install scipy: pip install scipy")
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Import Real_FITS model
from models.Real_FITS import Model as RealFITS


class Config:
    # Minimal config object for Real_FITS
    def __init__(self, seq_len, pred_len, enc_in, individual, cut_freq):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.individual = individual
        self.cut_freq = cut_freq


class ECGWindowDataset(Dataset):
    def __init__(self, signal, seq_len, pred_len=0, step=256):
        self.signal = signal.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.step = step
        self.indices = []
        max_start = len(signal) - (seq_len + pred_len)
        for start in range(0, max_start, step):
            self.indices.append(start)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.indices[idx]
        window = self.signal[s:s + self.seq_len]  # pred_len=0 (reconstruction)
        x = torch.from_numpy(window).unsqueeze(-1)  # (L, 1)
        return x, x  # input, target (reconstruction)


def load_edf_signal(edf_path, channel=0):
    f = pyedflib.EdfReader(edf_path)
    n = f.signals_in_file
    if channel >= n:
        raise ValueError(f"Requested channel {channel} but file has {n} channels")
    fs = f.getSampleFrequency(channel)
    sig = f.readSignal(channel)
    f._close()
    return sig, fs


def load_qrs_annotations(qrs_path, encoding='auto'):
    """
    Reads QRS annotation file (one sample index per line).
    Tries multiple encodings if 'auto' is given. Falls back to ignoring undecodable bytes.
    """
    enc_candidates = ['utf-8', 'cp1252', 'latin-1']
    if encoding != 'auto':
        enc_candidates = [encoding] + [e for e in enc_candidates if e != encoding]

    lines = None
    for enc in enc_candidates:
        try:
            with open(qrs_path, 'r', encoding=enc, errors='strict') as f:
                lines = f.readlines()
            # success
            break
        except Exception:
            continue

    if lines is None:
        # Fallback: binary read and ignore errors
        with open(qrs_path, 'rb') as f:
            raw = f.read().splitlines()
        lines = [r.decode('utf-8', 'ignore') for r in raw]

    peaks = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        tok = line.split()
        try:
            peaks.append(int(tok[0]))
        except ValueError:
            continue
    return np.array(sorted(set(peaks)), dtype=int)


def build_peak_label_array(length, peak_indices):
    labels = np.zeros(length, dtype=np.int32)
    valid = peak_indices[(peak_indices >= 0) & (peak_indices < length)]
    labels[valid] = 1
    return labels


def reconstruct_full_signal(model, signal, seq_len, pred_len, step, device):
    model.eval()
    length = len(signal)
    recon = np.zeros(length, dtype=np.float64)
    counts = np.zeros(length, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, length - (seq_len + pred_len) + 1, step):
            window = torch.from_numpy(signal[start:start + seq_len].astype(np.float32)).unsqueeze(0).unsqueeze(-1).to(device)
            out = model(window)  # shape (1, seq_len, 1) with pred_len=0
            if isinstance(out, tuple):
                out = out[0]
            out_win = out.squeeze(0).squeeze(-1).cpu().numpy()
            recon[start:start + seq_len] += out_win
            counts[start:start + seq_len] += 1.0
    mask = counts > 0
    recon[mask] /= counts[mask]
    # For any tail region not covered (if stride >1), fallback to original
    recon[~mask] = signal[~mask]
    return recon.astype(np.float32)


def auto_peak_height(signal, percentile=90):
    return np.percentile(signal, percentile)


def match_peaks(true_peaks, pred_peaks, tolerance):
    true_peaks = np.array(sorted(true_peaks))
    pred_peaks = np.array(sorted(pred_peaks))
    tp = 0
    matched_true = set()
    matched_pred = set()
    i = j = 0
    while i < len(true_peaks) and j < len(pred_peaks):
        if abs(true_peaks[i] - pred_peaks[j]) <= tolerance:
            tp += 1
            matched_true.add(i)
            matched_pred.add(j)
            i += 1
            j += 1
        elif pred_peaks[j] < true_peaks[i]:
            j += 1
        else:
            i += 1
    fp = len(pred_peaks) - len(matched_pred)
    fn = len(true_peaks) - len(matched_true)
    return tp, fp, fn


def compute_metrics_sample_level(true_labels, pred_labels):
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary', zero_division=0)
    acc = accuracy_score(true_labels, pred_labels)
    return acc, precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description="Fetal ECG peak detection with Real_FITS (MAE reconstruction)")
    parser.add_argument('--edf', type=str, default='addb/r01.edf')
    parser.add_argument('--qrs', type=str, default='addb/r01.edf.qrs')
    parser.add_argument('--channel', type=int, default=0)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--pred_len', type=int, default=0, help="Set 0 for pure reconstruction")
    parser.add_argument('--cut_ratio', type=float, default=1/24, help="cut_freq = int(seq_len * cut_ratio)")
    parser.add_argument('--individual', action='store_true', default=False)
    parser.add_argument('--stride', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--peak_percentile', type=float, default=90.0)
    parser.add_argument('--min_rr_sec', type=float, default=0.25, help="Minimum distance between fetal peaks (seconds)")
    parser.add_argument('--tolerance_ms', type=float, default=40.0, help="Matching tolerance in milliseconds")
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--out_plot', type=str, default='reconstruction_peaks.png')
    parser.add_argument('--qrs_encoding', type=str, default='auto', help="Encoding for QRS file or 'auto'")
    args = parser.parse_args()

    # Load data
    signal, fs = load_edf_signal(args.edf, args.channel)
    signal = signal.astype(np.float32)
    qrs_indices = load_qrs_annotations(args.qrs, encoding=args.qrs_encoding)
    labels = build_peak_label_array(len(signal), qrs_indices)

    # Normalize (z-score)
    mean = np.mean(signal)
    std = np.std(signal) + 1e-8
    norm_signal = (signal - mean) / std

    # Dataset / Loader
    train_dataset = ECGWindowDataset(norm_signal, seq_len=args.seq_len, pred_len=args.pred_len, step=args.stride)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Model config
    cut_freq = max(8, int(args.seq_len * args.cut_ratio))
    cfg = Config(seq_len=args.seq_len, pred_len=args.pred_len, enc_in=1, individual=args.individual, cut_freq=cut_freq)
    model = RealFITS(cfg).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()  # MAE

    # Training
    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for x, y in train_loader:
            x = x.to(args.device)  # (B, L, 1)
            y = y.to(args.device)
            optimizer.zero_grad()
            out = model(x)  # expect (B, L, 1) with pred_len=0
            if isinstance(out, tuple):
                out = out[0]
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch}/{args.epochs} - MAE: {epoch_loss:.6f}")

    # Reconstruction
    recon_norm = reconstruct_full_signal(model, norm_signal, args.seq_len, args.pred_len, args.stride, args.device)
    recon = recon_norm * std + mean

    # Peak detection on reconstructed signal
    height_thr = auto_peak_height(recon, args.peak_percentile)
    min_distance = int(fs * args.min_rr_sec)
    peaks_pred, _ = find_peaks(recon, height=height_thr, distance=min_distance)

    # Build predicted label array
    pred_labels = np.zeros(len(signal), dtype=np.int32)
    pred_labels[peaks_pred] = 1

    # Peak-level metrics
    tolerance_samples = int(fs * (args.tolerance_ms / 1000.0))
    tp, fp, fn = match_peaks(qrs_indices, peaks_pred, tolerance_samples)
    precision_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_p = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_p = 2 * precision_p * recall_p / (precision_p + recall_p) if (precision_p + recall_p) > 0 else 0.0
    accuracy_p = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0  # peak-wise accuracy (balanced)

    print("Peak-level metrics (tolerance {} samples):".format(tolerance_samples))
    print(f"TP {tp} FP {fp} FN {fn}")
    print(f"Precision {precision_p:.4f} Recall {recall_p:.4f} F1 {f1_p:.4f} Accuracy {accuracy_p:.4f}")

    # Sample-level metrics
    acc_s, prec_s, rec_s, f1_s = compute_metrics_sample_level(labels, pred_labels)
    print("Sample-level metrics:")
    print(f"Accuracy {acc_s:.4f} Precision {prec_s:.4f} Recall {rec_s:.4f} F1 {f1_s:.4f}")

    if args.plot:
        plt.figure(figsize=(14, 5))
        t = np.arange(len(signal)) / fs
        plt.plot(t, signal, label='Original', linewidth=0.8)
        plt.plot(t, recon, label='Reconstructed', linewidth=0.8, alpha=0.7)
        plt.scatter(qrs_indices / fs, signal[qrs_indices], color='green', s=18, label='True Peaks')
        plt.scatter(peaks_pred / fs, signal[peaks_pred], color='red', s=12, marker='x', label='Pred Peaks')
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("ECG Reconstruction & Peak Detection")
        plt.tight_layout()
        plt.savefig(args.out_plot, dpi=150)
        print(f"Saved plot to {args.out_plot}")


if __name__ == "__main__":
    main()
