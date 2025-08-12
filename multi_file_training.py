import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# WFDB library for proper annotation reading
import wfdb

try:
    import pyedflib
except ImportError:
    raise ImportError("Please install pyedflib: pip install pyedflib")

try:
    from scipy.signal import find_peaks
except ImportError:
    raise ImportError("Please install scipy: pip install scipy")

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from models.Real_FITS import Model as RealFITS

class Config:
    def __init__(self, seq_len, pred_len, enc_in, individual, cut_freq, **kwargs):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.individual = individual
        self.cut_freq = cut_freq
        
        # Data preprocessing options
        self.features = kwargs.get('features', 'S')
        self.target = kwargs.get('target', 'signal')
        self.scale = kwargs.get('scale', True)
        self.timeenc = kwargs.get('timeenc', 0)
        self.freq = kwargs.get('freq', 'h')
        
        # Data augmentation options
        self.in_dataset_augmentation = kwargs.get('in_dataset_augmentation', False)
        self.aug_method = kwargs.get('aug_method', 'noise')
        self.aug_rate = kwargs.get('aug_rate', 0.1)
        self.aug_data_size = kwargs.get('aug_data_size', 1)
        self.closer_data_aug_more = kwargs.get('closer_data_aug_more', False)
        
        # Data size control
        self.data_size = kwargs.get('data_size', 1.0)
        self.test_time_train = kwargs.get('test_time_train', False)

class MultiFileECGDataset(Dataset):
    """Enhanced ECG Dataset that can handle multiple files"""
    
    def __init__(self, config, file_data_list, flag='train', step=256):
        self.args = config
        self.file_data_list = file_data_list
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.step = step
        self.flag = flag
        
        # Set type mapping
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map.get(flag, 0)
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Process all file data
        self.__read_data__()
        self.collect_all_data()
        
        # Apply data augmentation for training set
        if self.args.in_dataset_augmentation and self.set_type == 0:
            self.data_augmentation()
    
    def __read_data__(self):
        """Process and normalize signals from all files"""
        all_signals = []
        self.processed_files = []
        
        for signal, qrs_indices, file_id in self.file_data_list:
            # Convert signal to proper format
            signal = signal.astype(np.float32)
            all_signals.append(signal.reshape(-1, 1))
            
            self.processed_files.append({
                'signal': signal,
                'qrs_indices': qrs_indices,
                'file_id': file_id,
                'length': len(signal)
            })
        
        # Concatenate all signals for global scaling
        if self.args.scale:
            combined_data = np.concatenate(all_signals, axis=0)
            self.scaler.fit(combined_data)
            
            # Scale each file's signal
            for i, file_info in enumerate(self.processed_files):
                scaled_signal = self.scaler.transform(file_info['signal'].reshape(-1, 1)).flatten()
                self.processed_files[i]['scaled_signal'] = scaled_signal
        else:
            for i, file_info in enumerate(self.processed_files):
                self.processed_files[i]['scaled_signal'] = file_info['signal']
    
    def collect_all_data(self):
        """Collect windowed data samples from all files"""
        self.x_data = []
        self.y_data = []
        self.file_ids = []
        
        for file_info in self.processed_files:
            signal = file_info['scaled_signal']
            file_id = file_info['file_id']
            
            # Calculate data length and masking
            data_len = len(signal) - self.seq_len - self.pred_len + 1
            mask_data_len = int((1 - self.args.data_size) * data_len) if self.args.data_size < 1 else 0
            
            # Create windows for this file
            for i in range(0, data_len, self.step):
                if (self.set_type == 0 and i >= mask_data_len) or self.set_type != 0:
                    s_begin = i
                    s_end = s_begin + self.seq_len
                    
                    if s_end <= len(signal):
                        self.x_data.append(signal[s_begin:s_end])
                        self.y_data.append(signal[s_begin:s_end])  # For reconstruction
                        self.file_ids.append(file_id)
    
    def data_augmentation(self):
        """Apply data augmentation"""
        origin_len = len(self.x_data)
        
        if not self.args.closer_data_aug_more:
            aug_size = [self.args.aug_data_size for i in range(origin_len)]
        else:
            aug_size = [int(self.args.aug_data_size * i / origin_len) + 1 for i in range(origin_len)]
        
        for i in range(origin_len):
            for _ in range(aug_size[i]):
                if self.args.aug_method == 'noise':
                    noise_factor = self.args.aug_rate
                    x_aug = self.x_data[i] + np.random.normal(0, noise_factor * np.std(self.x_data[i]), len(self.x_data[i]))
                    y_aug = self.y_data[i] + np.random.normal(0, noise_factor * np.std(self.y_data[i]), len(self.y_data[i]))
                    
                    self.x_data.append(x_aug)
                    self.y_data.append(y_aug)
                    self.file_ids.append(self.file_ids[i])
    
    def __getitem__(self, index):
        seq_x = torch.from_numpy(np.array(self.x_data[index])).float().unsqueeze(-1)
        seq_y = torch.from_numpy(np.array(self.y_data[index])).float().unsqueeze(-1)
        return seq_x, seq_y
    
    def __len__(self):
        return len(self.x_data)
    
    def inverse_transform(self, data):
        """Inverse transform the scaled data"""
        if self.args.scale:
            return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
        return data

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

def load_wfdb_annotations(record_name, data_dir, extension='qrs'):
    """Load QRS annotations using WFDB library with multiple format support"""
    print(f"Loading WFDB annotations for {record_name}.{extension}")
    
    # Construct full record path without extension
    record_path = os.path.join(data_dir, record_name)
    annotation_path = f"{record_path}.{extension}"
    
    print(f"Record path: {record_path}")
    print(f"Annotation file: {annotation_path}")
    
    # Check if annotation file exists
    if not os.path.exists(annotation_path):
        print(f"Annotation file not found: {annotation_path}")
        return np.array([])
    
    try:
        # Method 1: Use WFDB rdann function (most robust)
        print("Attempting WFDB rdann...")
        annotation = wfdb.rdann(record_path, extension)
        
        # Filter for QRS-related annotations
        qrs_symbols = ['N', 'L', 'R', 'V', 'A', 'F', 'J', 'E', 'j', 'Q', '/', 'f']
        
        # Get sample indices for QRS annotations
        qrs_indices = []
        for i, symbol in enumerate(annotation.symbol):
            if symbol in qrs_symbols:
                qrs_indices.append(annotation.sample[i])
        
        qrs_indices = np.array(qrs_indices, dtype=int)
        
        print(f"WFDB rdann successful: {len(qrs_indices)} QRS annotations")
        print(f"Annotation symbols found: {set(annotation.symbol)}")
        print(f"QRS symbols used: {[s for s in set(annotation.symbol) if s in qrs_symbols]}")
        
        return qrs_indices
        
    except Exception as e:
        print(f"WFDB rdann failed: {e}")
        
        # Method 2: Try reading as text-based QRS file
        try:
            print("Attempting text-based QRS parsing...")
            return load_text_qrs_annotations(annotation_path)
            
        except Exception as e2:
            print(f"Text parsing failed: {e2}")
            
            # Method 3: Try different common annotation extensions
            common_extensions = ['atr', 'qrs', 'ecg', 'ann']
            for ext in common_extensions:
                if ext != extension:
                    try:
                        print(f"Trying extension: {ext}")
                        annotation = wfdb.rdann(record_path, ext)
                        
                        qrs_indices = []
                        qrs_symbols = ['N', 'L', 'R', 'V', 'A', 'F', 'J', 'E', 'j', 'Q', '/', 'f']
                        for i, symbol in enumerate(annotation.symbol):
                            if symbol in qrs_symbols:
                                qrs_indices.append(annotation.sample[i])
                        
                        if len(qrs_indices) > 0:
                            print(f"Success with extension {ext}: {len(qrs_indices)} QRS annotations")
                            return np.array(qrs_indices, dtype=int)
                            
                    except:
                        continue
            
            print("All WFDB methods failed, returning empty array")
            return np.array([])

def load_text_qrs_annotations(qrs_path, encoding='auto'):
    """Fallback text-based QRS annotation loader"""
    print(f"Parsing text-based QRS file: {qrs_path}")
    
    enc_candidates = ['utf-8', 'cp1252', 'latin-1', 'ascii']
    if encoding != 'auto':
        enc_candidates = [encoding] + [e for e in enc_candidates if e != encoding]
    
    lines = None
    for enc in enc_candidates:
        try:
            with open(qrs_path, 'r', encoding=enc, errors='strict') as f:
                lines = f.readlines()
            print(f"Successfully read with encoding: {enc}")
            break
        except Exception as e:
            print(f"Failed with encoding {enc}: {e}")
            continue
    
    if lines is None:
        # Binary fallback
        with open(qrs_path, 'rb') as f:
            raw = f.read().splitlines()
        lines = [r.decode('utf-8', 'ignore') for r in raw]
        print("Binary read successful")
    
    print(f"Read {len(lines)} lines")
    
    # Parse peak indices
    peaks = []
    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        tokens = line.split()
        if not tokens:
            continue
            
        try:
            peak_idx = int(tokens[0])  # First token should be peak index
            peaks.append(peak_idx)
            
            if len(peaks) <= 3:  # Show first few
                print(f"   Line {line_num+1}: '{line}' -> Peak: {peak_idx}")
                
        except ValueError:
            if len(peaks) < 5:
                print(f"Could not parse line {line_num+1}: '{line}'")
            continue
    
    print(f"Parsed {len(peaks)} QRS peaks from text file")
    return np.array(sorted(set(peaks)), dtype=int)

def generate_synthetic_qrs(signal, fs, avg_hr=120):
    """Generate synthetic QRS peaks when annotations are missing"""
    avg_interval = int(fs * 60 / avg_hr)
    
    intervals = []
    pos = avg_interval
    while pos < len(signal):
        intervals.append(pos)
        next_interval = avg_interval + np.random.randint(-avg_interval//5, avg_interval//5)
        pos += next_interval
    
    return np.array(intervals)

def load_multiple_files_wfdb(file_indices, data_dir, channel=0):
    """Load multiple ECG files using WFDB with corrected file naming"""
    print(f"Loading multiple files using WFDB library...")
    print(f"Data directory: {data_dir}")
    print(f"File indices: {file_indices}")
    
    file_data_list = []
    fs = None
    
    for idx in file_indices:
        print(f"\n--- Processing File {idx} ---")
        
        try:
            # Use consistent double-digit naming (r01, r04, etc.)
            record_name = f'r{idx:02d}'
            
            # Load EDF signal
            signal_path = os.path.join(data_dir, f'{record_name}.edf')
            print(f"Loading signal from: {signal_path}")
            
            if not os.path.exists(signal_path):
                print(f"Signal file not found: {signal_path}")
                continue
                
            signal, current_fs = load_edf_signal(signal_path, channel)
            
            if fs is None:
                fs = current_fs
            elif fs != current_fs:
                print(f"Warning: Sampling rate mismatch. Expected {fs}, got {current_fs}")
            
            # Load annotations using WFDB
            qrs_indices = load_wfdb_annotations(record_name, data_dir, 'qrs')
            
            # Fallback to synthetic if no annotations found
            if len(qrs_indices) == 0:
                print(f"Generating synthetic QRS peaks for file {idx}")
                qrs_indices = generate_synthetic_qrs(signal, current_fs)
            
            file_data_list.append((signal, qrs_indices, idx))
            print(f"File {idx}: {len(signal)} samples, {len(qrs_indices)} QRS peaks")
            
        except Exception as e:
            print(f"Error loading file {idx}: {e}")
            continue
    
    print(f"\nSuccessfully loaded {len(file_data_list)} files")
    return file_data_list, fs

def show_wfdb_info(data_dir, record_name):
    """Display detailed WFDB information about a record"""
    print(f"\nWFDB Information for {record_name}")
    print("=" * 50)
    
    record_path = os.path.join(data_dir, record_name)
    
    try:
        # Try to read header
        header = wfdb.rdheader(record_path)
        print(f"Header Information:")
        print(f"   Record name: {header.record_name}")
        print(f"   Number of signals: {header.n_sig}")
        print(f"   Sampling frequency: {header.fs}")
        print(f"   Signal length: {header.sig_len}")
        print(f"   Signal names: {header.sig_name}")
        print(f"   Units: {header.units}")
        print(f"   File formats: {header.fmt}")
        
    except Exception as e:
        print(f"Could not read header: {e}")
    
    # Check for annotation files
    common_extensions = ['qrs', 'atr', 'ecg', 'ann']
    print(f"\nAvailable annotation files:")
    for ext in common_extensions:
        ann_path = f"{record_path}.{ext}"
        if os.path.exists(ann_path):
            try:
                annotation = wfdb.rdann(record_path, ext)
                print(f"   {ext}: {len(annotation.sample)} annotations")
                print(f"      Symbols: {set(annotation.symbol)}")
            except Exception as e:
                print(f"   {ext}: File exists but could not read - {e}")
        else:
            print(f"   {ext}: Not found")

def split_files_for_training(file_data_list, test_ratio=0.2):
    """Split files into train and test sets"""
    n_files = len(file_data_list)
    n_test = max(1, int(n_files * test_ratio))
    
    indices = np.random.permutation(n_files)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    train_files = [file_data_list[i] for i in train_indices]
    test_files = [file_data_list[i] for i in test_indices]
    
    return train_files, test_files

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

def evaluate_model(model, dataset, device, original_files, fs, args):
    """Evaluate model on test dataset"""
    model.eval()
    all_results = []
    
    # Group data by file ID for evaluation
    file_groups = {}
    for i, file_id in enumerate(dataset.file_ids):
        if file_id not in file_groups:
            file_groups[file_id] = []
        file_groups[file_id].append(i)
    
    # Evaluate each file separately
    for file_id, indices in file_groups.items():
        print(f"\nEvaluating file {file_id}...")
        
        # Find original file data
        original_file = None
        for signal, qrs_indices, fid in original_files:
            if fid == file_id:
                original_file = (signal, qrs_indices)
                break
        
        if original_file is None:
            continue
            
        original_signal, true_qrs = original_file
        
        # Reconstruct signal for this file
        recon_signal = np.zeros_like(original_signal, dtype=np.float32)
        counts = np.zeros_like(original_signal, dtype=np.float32)
        
        with torch.no_grad():
            for idx in indices:
                seq_x, _ = dataset[idx]
                seq_x = seq_x.unsqueeze(0).to(device)
                
                out = model(seq_x)
                if isinstance(out, tuple):
                    out = out[0]
                
                recon_seq = out.squeeze(0).squeeze(-1).cpu().numpy()
                
                # Calculate position in original signal
                start_pos = idx * dataset.step
                end_pos = start_pos + dataset.seq_len
                
                if end_pos <= len(recon_signal):
                    recon_signal[start_pos:end_pos] += recon_seq
                    counts[start_pos:end_pos] += 1.0
        
        # Average overlapping regions and inverse transform
        mask = counts > 0
        recon_signal[mask] /= counts[mask]
        recon_signal = dataset.inverse_transform(recon_signal)
        
        # Peak detection
        height_thr = np.percentile(recon_signal, args.peak_percentile)
        min_distance = int(fs * args.min_rr_sec)
        peaks_pred, _ = find_peaks(recon_signal, height=height_thr, distance=min_distance)
        
        # Calculate metrics
        tolerance_samples = int(fs * (args.tolerance_ms / 1000.0))
        tp, fp, fn = match_peaks(true_qrs, peaks_pred, tolerance_samples)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        result = {
            'file_id': file_id,
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1': f1,
            'true_peaks': len(true_qrs), 'pred_peaks': len(peaks_pred)
        }
        all_results.append(result)
        
        print(f"File {file_id} - TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return all_results

def summarize_results_across_folds(all_fold_results):
    """
    all_fold_results: list of dicts, each containing keys:
      - 'test_file'
      - 'avg_precision'
      - 'avg_recall'
      - 'avg_f1'
      - 'avg_accuracy_peak'
      - 'avg_accuracy_sample'
      - 'details' (the list returned from evaluate_model)
    """
    if not all_fold_results:
        print("No fold results to summarize.")
        return

    avg_precision = np.mean([fr['avg_precision'] for fr in all_fold_results])
    avg_recall = np.mean([fr['avg_recall'] for fr in all_fold_results])
    avg_f1 = np.mean([fr['avg_f1'] for fr in all_fold_results])
    avg_acc_peak = np.mean([fr.get('avg_accuracy_peak', 0.0) for fr in all_fold_results])
    avg_acc_sample = np.mean([fr.get('avg_accuracy_sample', 0.0) for fr in all_fold_results])

    print("\n===== LOSO Summary Across Folds =====")
    print(f"Folds: {len(all_fold_results)}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall:    {avg_recall:.4f}")
    print(f"Average F1:        {avg_f1:.4f}")
    print(f"Average Peak Acc:  {avg_acc_peak:.4f}")
    print(f"Average Sample Acc:{avg_acc_sample:.4f}")

    print("\nPer-fold results:")
    for fr in all_fold_results:
        print(f"- Test file {fr['test_file']}: "
              f"P={fr['avg_precision']:.3f}, R={fr['avg_recall']:.3f}, "
              f"F1={fr['avg_f1']:.3f}, Acc_peak={fr['avg_accuracy_peak']:.3f}, "
              f"Acc_sample={fr['avg_accuracy_sample']:.3f}")

def run_loso(args):
    """
    Perform Leave-One-Subject-Out CV across provided file_indices.
    For each held-out file:
      - Train on the remaining files
      - Test on the held-out file
    Aggregates metrics across all folds.
    """
    print("Running LOSO cross-validation...")
    print(f"Subjects (files): {args.file_indices}")
    # Load all files once
    file_data_list, fs = load_multiple_files_wfdb(args.file_indices, args.data_dir, args.channel)
    if len(file_data_list) < 2:
        print("Need at least 2 files for LOSO.")
        return

    # Map: file_id -> (signal, qrs, id)
    by_id = {fid: (sig, qrs, fid) for (sig, qrs, fid) in file_data_list}

    all_fold_results = []

    for test_id in args.file_indices:
        if test_id not in by_id:
            print(f"Skipping LOSO fold for {test_id}, not loaded.")
            continue

        print("\n" + "="*60)
        print(f"LOSO fold: Test on file {test_id}, Train on the rest")
        print("="*60)

        # Build train/test lists for this fold
        test_files = [by_id[test_id]]
        train_files = [by_id[fid] for fid in args.file_indices if fid != test_id and fid in by_id]

        if len(train_files) == 0:
            print(f"Skipping fold for {test_id}: no training files available.")
            continue

        # Build config
        cut_freq = max(8, int(args.seq_len * args.cut_ratio))
        config = Config(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=1,
            individual=args.individual,
            cut_freq=cut_freq,
            scale=args.scale,
            data_size=args.data_size,
            in_dataset_augmentation=args.in_dataset_augmentation,
            aug_method=args.aug_method,
            aug_rate=args.aug_rate,
            aug_data_size=args.aug_data_size
        )

        # Datasets and loaders
        print("Creating datasets for this fold...")
        train_dataset = MultiFileECGDataset(config, train_files, flag='train', step=args.stride)
        test_dataset = MultiFileECGDataset(config, test_files, flag='test', step=args.stride)

        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  drop_last=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)

        # Model/init
        print("Initializing model for this fold...")
        model = RealFITS(config).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.L1Loss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        best_loss = float('inf')
        patience_counter = 0

        print(f"Starting training for up to {args.epochs} epochs...")
        for epoch in range(1, args.epochs + 1):
            # Train
            model.train()
            epoch_train_loss = 0.0
            train_batches = 0
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

                epoch_train_loss += loss.item()
                train_batches += 1

            avg_train_loss = epoch_train_loss / max(1, train_batches)

            # Validate on test set for early stopping
            model.eval()
            epoch_val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(args.device)
                    y = y.to(args.device)
                    out = model(x)
                    if isinstance(out, tuple):
                        out = out[0]
                    loss = criterion(out, y)
                    epoch_val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = epoch_val_loss / max(1, val_batches)
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), args.save_model)
                print(f"Fold Test={test_id} Epoch {epoch}/{args.epochs} "
                      f"- Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f} [BEST]")
            else:
                patience_counter += 1
                print(f"Fold Test={test_id} Epoch {epoch}/{args.epochs} "
                      f"- Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping for fold Test={test_id} at epoch {epoch}")
                break

        # Evaluate best model on the held-out file
        print("Loading best model for evaluation for this fold...")
        model.load_state_dict(torch.load(args.save_model))

        print("Evaluating on held-out test file...")
        results = evaluate_model(model, test_dataset, args.device, test_files, fs, args)

        # Fold summary
        if results:
            avg_precision = np.mean([r['precision'] for r in results])
            avg_recall = np.mean([r['recall'] for r in results])
            avg_f1 = np.mean([r['f1'] for r in results])
            avg_accuracy_peak = np.mean([r.get('accuracy_peak', 0.0) for r in results])
            avg_accuracy_sample = np.mean([r.get('accuracy_sample', 0.0) for r in results])

            print("\nFold summary:")
            print(f"Test file: {test_id}")
            print(f"Precision: {avg_precision:.4f}")
            print(f"Recall:    {avg_recall:.4f}")
            print(f"F1:        {avg_f1:.4f}")
            print(f"Acc_peak:  {avg_accuracy_peak:.4f}")
            print(f"Acc_sample:{avg_accuracy_sample:.4f}")

            all_fold_results.append({
                'test_file': test_id,
                'avg_precision': avg_precision,
                'avg_recall': avg_recall,
                'avg_f1': avg_f1,
                'avg_accuracy_peak': avg_accuracy_peak,
                'avg_accuracy_sample': avg_accuracy_sample,
                'details': results
            })

    # Global LOSO summary
    summarize_results_across_folds(all_fold_results)


def compute_metrics_sample_level(true_labels, pred_labels):
    """Compute sample-level metrics including accuracy"""
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='binary', zero_division=0
    )
    acc = accuracy_score(true_labels, pred_labels)
    return acc, precision, recall, f1

def evaluate_model(model, dataset, device, original_files, fs, args):
    """Enhanced evaluation with both peak-level and sample-level accuracy"""
    model.eval()
    all_results = []
    
    # Group data by file ID for evaluation
    file_groups = {}
    for i, file_id in enumerate(dataset.file_ids):
        if file_id not in file_groups:
            file_groups[file_id] = []
        file_groups[file_id].append(i)
    
    # Evaluate each file separately
    for file_id, indices in file_groups.items():
        print(f"\nEvaluating file {file_id}...")
        
        # Find original file data
        original_file = None
        for signal, qrs_indices, fid in original_files:
            if fid == file_id:
                original_file = (signal, qrs_indices)
                break
        
        if original_file is None:
            continue
            
        original_signal, true_qrs = original_file
        
        # Reconstruct signal for this file
        recon_signal = np.zeros_like(original_signal, dtype=np.float32)
        counts = np.zeros_like(original_signal, dtype=np.float32)
        
        with torch.no_grad():
            for idx in indices:
                seq_x, _ = dataset[idx]
                seq_x = seq_x.unsqueeze(0).to(device)
                
                out = model(seq_x)
                if isinstance(out, tuple):
                    out = out[0]
                
                recon_seq = out.squeeze(0).squeeze(-1).cpu().numpy()
                
                # Calculate position in original signal
                start_pos = idx * dataset.step
                end_pos = start_pos + dataset.seq_len
                
                if end_pos <= len(recon_signal):
                    recon_signal[start_pos:end_pos] += recon_seq
                    counts[start_pos:end_pos] += 1.0
        
        # Average overlapping regions and inverse transform
        mask = counts > 0
        recon_signal[mask] /= counts[mask]
        recon_signal = dataset.inverse_transform(recon_signal)
        
        # Peak detection
        height_thr = np.percentile(recon_signal, args.peak_percentile)
        min_distance = int(fs * args.min_rr_sec)
        peaks_pred, _ = find_peaks(recon_signal, height=height_thr, distance=min_distance)
        
        # Peak-level metrics
        tolerance_samples = int(fs * (args.tolerance_ms / 1000.0))
        tp, fp, fn = match_peaks(true_qrs, peaks_pred, tolerance_samples)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # ADD PEAK-LEVEL ACCURACY
        accuracy_peak = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
        # Sample-level metrics
        true_labels = np.zeros(len(original_signal), dtype=np.int32)
        true_labels[true_qrs] = 1
        
        pred_labels = np.zeros(len(original_signal), dtype=np.int32)
        valid_peaks = peaks_pred[peaks_pred < len(original_signal)]
        pred_labels[valid_peaks] = 1
        
        # ADD SAMPLE-LEVEL ACCURACY
        accuracy_sample, precision_sample, recall_sample, f1_sample = compute_metrics_sample_level(
            true_labels, pred_labels
        )
        
        result = {
            'file_id': file_id,
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1': f1,
            'accuracy_peak': accuracy_peak,  # NEW
            'accuracy_sample': accuracy_sample,  # NEW
            'precision_sample': precision_sample,  # NEW
            'recall_sample': recall_sample,  # NEW
            'f1_sample': f1_sample,  # NEW
            'true_peaks': len(true_qrs), 'pred_peaks': len(peaks_pred)
        }
        all_results.append(result)
        
        # ENHANCED PRINTING WITH ACCURACY
        print(f"File {file_id} - TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"Peak-level    - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy_peak:.4f}")
        print(f"Sample-level  - Precision: {precision_sample:.4f}, Recall: {recall_sample:.4f}, F1: {f1_sample:.4f}, Accuracy: {accuracy_sample:.4f}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Multi-file ECG training with Real_FITS")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing ECG files')
    parser.add_argument('--file_indices', type=int, nargs='+', default=[1, 4, 7, 8, 10], help='File indices to use')
    parser.add_argument('--channel', type=int, default=0)
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Ratio of files to use for testing')
    parser.add_argument('--show_info', action='store_true', help='Show detailed WFDB info')
    parser.add_argument('--loso', action='store_true', help='Run LOSO cross-validation across provided file_indices')

    
    # Model arguments
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--pred_len', type=int, default=0)
    parser.add_argument('--cut_ratio', type=float, default=1/24)
    parser.add_argument('--individual', action='store_true', default=False)
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--stride', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # Data preprocessing
    parser.add_argument('--scale', action='store_true', default=True)
    parser.add_argument('--data_size', type=float, default=1.0)
    parser.add_argument('--in_dataset_augmentation', action='store_true', default=False)
    parser.add_argument('--aug_method', type=str, default='noise')
    parser.add_argument('--aug_rate', type=float, default=0.1)
    parser.add_argument('--aug_data_size', type=int, default=1)
    
    # Evaluation arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--peak_percentile', type=float, default=90.0)
    parser.add_argument('--min_rr_sec', type=float, default=0.25)
    parser.add_argument('--tolerance_ms', type=float, default=40.0)
    
    # Output arguments
    parser.add_argument('--save_model', type=str, default='best_model.pth')
    parser.add_argument('--early_stopping_patience', type=int, default=20)
    
    args = parser.parse_args()
    
    print(f"Enhanced ECG Training with WFDB Annotation Support")
    print(f"Using files: {args.file_indices}")
    print(f"Device: {args.device}")
    
    # Show detailed info if requested
    if args.show_info:
        for idx in args.file_indices:
            show_wfdb_info(args.data_dir, f'r{idx:02d}')
    
    if args.loso:
    # Run LOSO and exit
        run_loso(args)
        return
    
    # Load multiple files using WFDB
    print("\nLoading ECG files with WFDB...")
    file_data_list, fs = load_multiple_files_wfdb(args.file_indices, args.data_dir, args.channel)
    
    if len(file_data_list) == 0:
        print("No files loaded successfully!")
        return
    
    print(f"Successfully loaded {len(file_data_list)} files")
    
    # Split files for training and testing
    train_files, test_files = split_files_for_training(file_data_list, args.test_ratio)
    print(f"Training files: {[f[2] for f in train_files]}")
    print(f"Test files: {[f[2] for f in test_files]}")
    
    # Create config
    cut_freq = max(8, int(args.seq_len * args.cut_ratio))
    config = Config(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        enc_in=1,
        individual=args.individual,
        cut_freq=cut_freq,
        scale=args.scale,
        data_size=args.data_size,
        in_dataset_augmentation=args.in_dataset_augmentation,
        aug_method=args.aug_method,
        aug_rate=args.aug_rate,
        aug_data_size=args.aug_data_size
    )
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = MultiFileECGDataset(config, train_files, flag='train', step=args.stride)
    test_dataset = MultiFileECGDataset(config, test_files, flag='test', step=args.stride)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Initialize model
    print("\nInitializing model...")
    model = RealFITS(config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop with early stopping
    print(f"\nStarting training for {args.epochs} epochs...")
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        epoch_train_loss = 0.0
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
            
            epoch_train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(args.device)
                y = y.to(args.device)
                
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                
                loss = criterion(out, y)
                epoch_val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = epoch_val_loss / val_batches if val_batches > 0 else float('inf')
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping and model saving
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), args.save_model)
            print(f"Epoch {epoch}/{args.epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f} [BEST]")
        else:
            patience_counter += 1
            print(f"Epoch {epoch}/{args.epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        if patience_counter >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model for evaluation
    print(f"\nLoading best model for evaluation...")
    model.load_state_dict(torch.load(args.save_model))
    
    # Evaluate on test files
    print("\nEvaluating on test files...")
    results = evaluate_model(model, test_dataset, args.device, test_files, fs, args)
    
    # Summary statistics
    if results:
        avg_precision = np.mean([r['precision'] for r in results])
        avg_recall = np.mean([r['recall'] for r in results])
        avg_f1 = np.mean([r['f1'] for r in results])
        avg_accuracy_peak = np.mean([r['accuracy_peak'] for r in results])
        avg_accuracy_sample = np.mean([r['accuracy_sample'] for r in results])
        
        print(f"\nOverall Peak-level Results:")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average F1: {avg_f1:.4f}")
        print(f"Average Accuracy: {avg_accuracy_peak:.4f}")
        
        print(f"\nOverall Sample-level Results:")
        print(f"Average Accuracy: {avg_accuracy_sample:.4f}")
        
        # Detailed results
        print(f"\nDetailed Results:")
        for result in results:
            print(f"File {result['file_id']}: P={result['precision']:.3f}, R={result['recall']:.3f}, F1={result['f1']:.3f}, Acc={result['accuracy_peak']:.3f}")

        
if __name__ == "__main__":
    main()