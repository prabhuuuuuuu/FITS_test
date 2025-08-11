import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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
    # Enhanced config object for Real_FITS with data preprocessing options
    def __init__(self, seq_len, pred_len, enc_in, individual, cut_freq, **kwargs):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.individual = individual
        self.cut_freq = cut_freq
        
        # Data preprocessing options
        self.features = kwargs.get('features', 'S')  # 'S': single feature, 'M': multivariate
        self.target = kwargs.get('target', 'signal')
        self.scale = kwargs.get('scale', True)
        self.timeenc = kwargs.get('timeenc', 0)
        self.freq = kwargs.get('freq', 'h')
        
        # Data augmentation options
        self.in_dataset_augmentation = kwargs.get('in_dataset_augmentation', False)
        self.aug_method = kwargs.get('aug_method', 'f_mask')
        self.aug_rate = kwargs.get('aug_rate', 0.1)
        self.aug_data_size = kwargs.get('aug_data_size', 1)
        self.closer_data_aug_more = kwargs.get('closer_data_aug_more', False)
        
        # Data size control
        self.data_size = kwargs.get('data_size', 1.0)
        self.test_time_train = kwargs.get('test_time_train', False)

class ECGDataset(Dataset):
    """Enhanced ECG Dataset with preprocessing capabilities from data_loader.py"""
    
    def __init__(self, config, signal, qrs_indices=None, flag='train', step=256):
        self.args = config
        self.signal = signal.astype(np.float32)
        self.qrs_indices = qrs_indices
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.step = step
        self.flag = flag
        
        # Set type mapping
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map.get(flag, 0)
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Process the signal data
        self.__read_data__()
        self.collect_all_data()
        
        # Apply data augmentation for training set
        if self.args.in_dataset_augmentation and self.set_type == 0:
            self.data_augmentation()
    
    def __read_data__(self):
        """Process and normalize the signal data"""
        # Convert signal to DataFrame-like structure
        data = self.signal.reshape(-1, 1)  # Shape: (length, 1)
        
        # Apply scaling if enabled
        if self.args.scale:
            self.scaler.fit(data)
            self.data_x = self.scaler.transform(data).flatten()
            self.data_y = self.data_x.copy()
        else:
            self.data_x = data.flatten()
            self.data_y = self.data_x.copy()
        
        # Create time stamps (placeholder for now)
        self.data_stamp = np.arange(len(self.data_x)).reshape(-1, 1)
    
    def collect_all_data(self):
        """Collect windowed data samples"""
        self.x_data = []
        self.y_data = []
        self.x_time = []
        self.y_time = []
        
        # Calculate data length and masking
        data_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        mask_data_len = int((1 - self.args.data_size) * data_len) if self.args.data_size < 1 else 0
        
        # Create windows
        for i in range(0, data_len, self.step):
            if (self.set_type == 0 and i >= mask_data_len) or self.set_type != 0:
                s_begin = i
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.seq_len  # For reconstruction, label starts from beginning
                r_end = r_begin + self.seq_len + self.pred_len
                
                # Ensure we don't go out of bounds
                if r_end <= len(self.data_x):
                    self.x_data.append(self.data_x[s_begin:s_end])
                    self.y_data.append(self.data_y[r_begin:r_begin + self.seq_len])  # For reconstruction
                    self.x_time.append(self.data_stamp[s_begin:s_end])
                    self.y_time.append(self.data_stamp[r_begin:r_begin + self.seq_len])
    
    def data_augmentation(self):
        """Apply data augmentation (placeholder - can be enhanced with frequency domain augmentation)"""
        origin_len = len(self.x_data)
        
        if not self.args.closer_data_aug_more:
            aug_size = [self.args.aug_data_size for i in range(origin_len)]
        else:
            aug_size = [int(self.args.aug_data_size * i / origin_len) + 1 for i in range(origin_len)]
        
        for i in range(origin_len):
            for _ in range(aug_size[i]):
                if self.args.aug_method == 'noise':
                    # Simple noise augmentation
                    noise_factor = self.args.aug_rate
                    x_aug = self.x_data[i] + np.random.normal(0, noise_factor * np.std(self.x_data[i]), len(self.x_data[i]))
                    y_aug = self.y_data[i] + np.random.normal(0, noise_factor * np.std(self.y_data[i]), len(self.y_data[i]))
                    
                    self.x_data.append(x_aug)
                    self.y_data.append(y_aug)
                    self.x_time.append(self.x_time[i])
                    self.y_time.append(self.y_time[i])
    
    def __getitem__(self, index):
        seq_x = torch.from_numpy(np.array(self.x_data[index])).float().unsqueeze(-1)  # (seq_len, 1)
        seq_y = torch.from_numpy(np.array(self.y_data[index])).float().unsqueeze(-1)  # (seq_len, 1)
        return seq_x, seq_y
    
    def __len__(self):
        return len(self.x_data)
    
    def inverse_transform(self, data):
        """Inverse transform the scaled data"""
        if self.args.scale:
            return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
        return data

def data_provider(args, signal, qrs_indices=None, flag='train'):
    """Data provider function similar to data_factory.py"""
    
    # Determine data loading parameters based on flag
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    else:  # train/val
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    
    # Create dataset
    data_set = ECGDataset(
        config=args,
        signal=signal,
        qrs_indices=qrs_indices,
        flag=flag,
        step=args.stride
    )
    
    print(f"{flag} dataset length: {len(data_set)}")
    
    # Create data loader
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=getattr(args, 'num_workers', 0),
        drop_last=drop_last
    )
    
    return data_set, data_loader

def load_edf_signal(edf_path, channel=0):
    """Load EDF signal"""
    f = pyedflib.EdfReader(edf_path)
    n = f.signals_in_file
    if channel >= n:
        raise ValueError(f"Requested channel {channel} but file has {n} channels")
    fs = f.getSampleFrequency(channel)
    sig = f.readSignal(channel)
    f._close()
    return sig, fs

def load_qrs_annotations(qrs_path, encoding='auto'):
    """Load QRS annotations"""
    enc_candidates = ['utf-8', 'cp1252', 'latin-1']
    if encoding != 'auto':
        enc_candidates = [encoding] + [e for e in enc_candidates if e != encoding]
    
    lines = None
    for enc in enc_candidates:
        try:
            with open(qrs_path, 'r', encoding=enc, errors='strict') as f:
                lines = f.readlines()
            break
        except Exception:
            continue
    
    if lines is None:
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
    """Build binary label array for peaks"""
    labels = np.zeros(length, dtype=np.int32)
    valid = peak_indices[(peak_indices >= 0) & (peak_indices < length)]
    labels[valid] = 1
    return labels

def reconstruct_full_signal(model, dataset, device):
    """Reconstruct full signal using trained model"""
    model.eval()
    
    # Get original signal length
    original_length = len(dataset.signal)
    recon = np.zeros(original_length, dtype=np.float64)
    counts = np.zeros(original_length, dtype=np.float64)
    
    # Create data loader for reconstruction
    recon_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for i, (x, _) in enumerate(recon_loader):
            x = x.to(device)  # (1, seq_len, 1)
            out = model(x)
            
            if isinstance(out, tuple):
                out = out[0]
            
            # Get reconstruction
            out_seq = out.squeeze(0).squeeze(-1).cpu().numpy()  # (seq_len,)
            
            # Calculate position in original signal
            start_idx = i * dataset.step
            end_idx = start_idx + dataset.seq_len
            
            if end_idx <= original_length:
                recon[start_idx:end_idx] += out_seq
                counts[start_idx:end_idx] += 1.0
    
    # Average overlapping regions
    mask = counts > 0
    recon[mask] /= counts[mask]
    
    # Inverse transform if scaling was applied
    if dataset.args.scale:
        recon = dataset.inverse_transform(recon)
    
    return recon.astype(np.float32)

def auto_peak_height(signal, percentile=90):
    """Automatically determine peak height threshold"""
    return np.percentile(signal, percentile)

def match_peaks(true_peaks, pred_peaks, tolerance):
    """Match predicted peaks with true peaks"""
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
    """Compute sample-level metrics"""
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='binary', zero_division=0
    )
    acc = accuracy_score(true_labels, pred_labels)
    return acc, precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description="Enhanced Fetal ECG peak detection with data preprocessing")
    
    # Data arguments
    parser.add_argument('--edf', type=str, default='addb/r01.edf')
    parser.add_argument('--qrs', type=str, default='addb/r01.edf.qrs')
    parser.add_argument('--channel', type=int, default=0)
    
    # Model arguments
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--pred_len', type=int, default=0, help="Set 0 for pure reconstruction")
    parser.add_argument('--cut_ratio', type=float, default=1/24)
    parser.add_argument('--individual', action='store_true', default=False)
    
    # Data preprocessing arguments
    parser.add_argument('--features', type=str, default='S', choices=['S', 'M', 'MS'])
    parser.add_argument('--scale', action='store_true', default=True)
    parser.add_argument('--data_size', type=float, default=1.0)
    
    # Data augmentation arguments
    parser.add_argument('--in_dataset_augmentation', action='store_true', default=False)
    parser.add_argument('--aug_method', type=str, default='noise', choices=['noise', 'f_mask'])
    parser.add_argument('--aug_rate', type=float, default=0.1)
    parser.add_argument('--aug_data_size', type=int, default=1)
    
    # Training arguments
    parser.add_argument('--stride', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # Evaluation arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--peak_percentile', type=float, default=90.0)
    parser.add_argument('--min_rr_sec', type=float, default=0.25)
    parser.add_argument('--tolerance_ms', type=float, default=40.0)
    
    # Output arguments
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--out_plot', type=str, default='reconstruction_peaks.png')
    parser.add_argument('--qrs_encoding', type=str, default='auto')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading EDF signal and QRS annotations...")
    signal, fs = load_edf_signal(args.edf, args.channel)
    signal = signal.astype(np.float32)
    qrs_indices = load_qrs_annotations(args.qrs, encoding=args.qrs_encoding)
    labels = build_peak_label_array(len(signal), qrs_indices)
    
    print(f"Signal length: {len(signal)}, Sampling rate: {fs} Hz")
    print(f"Number of QRS peaks: {len(qrs_indices)}")
    
    # Create enhanced config
    cut_freq = max(8, int(args.seq_len * args.cut_ratio))
    config = Config(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        enc_in=1,
        individual=args.individual,
        cut_freq=cut_freq,
        features=getattr(args, 'features', 'S'),
        scale=getattr(args, 'scale', True),
        data_size=getattr(args, 'data_size', 1.0),
        in_dataset_augmentation=getattr(args, 'in_dataset_augmentation', False),
        aug_method=getattr(args, 'aug_method', 'noise'),
        aug_rate=getattr(args, 'aug_rate', 0.1),
        aug_data_size=getattr(args, 'aug_data_size', 1)
    )
    
    # Create datasets and data loaders
    print("Creating datasets...")
    train_dataset, train_loader = data_provider(args, signal, qrs_indices, flag='train')
    
    # Initialize model
    print("Initializing model...")
    model = RealFITS(config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()  # MAE
    
    # Training
    print("Starting training...")
    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        
        for x, y in train_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            
            optimizer.zero_grad()
            out = model(x)
            
            if isinstance(out, tuple):
                out = out[0]
            
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch}/{args.epochs} - Average MAE: {avg_loss:.6f}")
    
    # Reconstruction and evaluation
    print("Reconstructing signal...")
    recon = reconstruct_full_signal(model, train_dataset, args.device)
    
    # Peak detection on reconstructed signal
    print("Detecting peaks...")
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
    accuracy_p = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    print(f"\nPeak-level metrics (tolerance {tolerance_samples} samples):")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"Precision: {precision_p:.4f}, Recall: {recall_p:.4f}, F1: {f1_p:.4f}, Accuracy: {accuracy_p:.4f}")
    
    # Sample-level metrics
    acc_s, prec_s, rec_s, f1_s = compute_metrics_sample_level(labels, pred_labels)
    print(f"\nSample-level metrics:")
    print(f"Accuracy: {acc_s:.4f}, Precision: {prec_s:.4f}, Recall: {rec_s:.4f}, F1: {f1_s:.4f}")
    
    # Plotting
    if args.plot:
        print(f"Creating plot...")
        plt.figure(figsize=(14, 5))
        t = np.arange(len(signal)) / fs
        
        plt.plot(t, signal, label='Original', linewidth=0.8)
        plt.plot(t, recon, label='Reconstructed', linewidth=0.8, alpha=0.7)
        plt.scatter(qrs_indices / fs, signal[qrs_indices], color='green', s=18, label='True Peaks')
        plt.scatter(peaks_pred / fs, signal[peaks_pred], color='red', s=12, marker='x', label='Pred Peaks')
        
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("ECG Reconstruction & Peak Detection with Enhanced Preprocessing")
        plt.tight_layout()
        plt.savefig(args.out_plot, dpi=150)
        print(f"Saved plot to {args.out_plot}")

if __name__ == "__main__":
    main()
