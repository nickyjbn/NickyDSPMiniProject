"""
Feature Extraction and Classification Module
Digital Signal Processing Mini-Project

Implements:
- Time-domain feature extraction from ECG
- Frequency-domain features from DFT/DCT/DWT
- Texture features from scalograms
- Simple classification (threshold-based and SVM)
"""

import numpy as np
from scipy import stats
from scipy.signal.windows import gaussian
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ECGFeatureExtractor:
    """
    Extract features from ECG signals for classification
    Combines time-domain, frequency-domain, and image-based features
    """
    
    def __init__(self, sampling_rate: float = 360.0):
        """
        Initialize feature extractor
        
        Args:
            sampling_rate: ECG sampling rate in Hz
        """
        self.fs = sampling_rate
        self.features = {}
        
    # =========================================================================
    # Time-Domain Features from 1D Signal
    # =========================================================================
    
    def extract_qrs_features(self, ecg_signal: np.ndarray, 
                            r_peaks: np.ndarray) -> Dict[str, float]:
        """
        Extract QRS complex features
        
        Features:
        - QRS width (duration)
        - QRS amplitude
        - QRS area
        
        Args:
            ecg_signal: Filtered ECG signal
            r_peaks: R-peak locations
            
        Returns:
            Dictionary of QRS features
        """
        qrs_widths = []
        qrs_amplitudes = []
        qrs_areas = []
        
        # Typical QRS duration: 60-100 ms
        qrs_window = int(0.1 * self.fs)  # 100ms window
        
        for r_peak in r_peaks:
            # Extract QRS segment
            start_idx = max(0, r_peak - qrs_window // 2)
            end_idx = min(len(ecg_signal), r_peak + qrs_window // 2)
            qrs_segment = ecg_signal[start_idx:end_idx]
            
            # QRS width (using threshold crossing)
            threshold = np.max(qrs_segment) * 0.5
            above_threshold = qrs_segment > threshold
            if np.any(above_threshold):
                width_samples = np.sum(above_threshold)
                width_ms = (width_samples / self.fs) * 1000
                qrs_widths.append(width_ms)
            
            # QRS amplitude (R-peak amplitude)
            amplitude = ecg_signal[r_peak]
            qrs_amplitudes.append(amplitude)
            
            # QRS area (integral)
            area = np.trapezoid(np.abs(qrs_segment))
            qrs_areas.append(area)
        
        features = {
            'qrs_width_mean_ms': np.mean(qrs_widths) if qrs_widths else 0,
            'qrs_width_std_ms': np.std(qrs_widths) if qrs_widths else 0,
            'qrs_amplitude_mean': np.mean(qrs_amplitudes) if qrs_amplitudes else 0,
            'qrs_amplitude_std': np.std(qrs_amplitudes) if qrs_amplitudes else 0,
            'qrs_area_mean': np.mean(qrs_areas) if qrs_areas else 0,
            'qrs_area_std': np.std(qrs_areas) if qrs_areas else 0,
        }
        
        print(f"[Feature Extraction] QRS features:")
        print(f"  Mean QRS width: {features['qrs_width_mean_ms']:.1f} ms")
        print(f"  Mean QRS amplitude: {features['qrs_amplitude_mean']:.3f}")
        
        return features
    
    def extract_st_segment_features(self, ecg_signal: np.ndarray, 
                                   r_peaks: np.ndarray) -> Dict[str, float]:
        """
        Extract ST segment features
        
        ST segment elevation/depression is important for detecting ischemia
        
        Args:
            ecg_signal: Filtered ECG signal
            r_peaks: R-peak locations
            
        Returns:
            Dictionary of ST segment features
        """
        st_elevations = []
        
        # ST segment starts ~80ms after R-peak
        st_start_offset = int(0.08 * self.fs)
        st_end_offset = int(0.12 * self.fs)
        
        for r_peak in r_peaks:
            # Baseline (isoelectric line) - take before QRS
            baseline_start = max(0, r_peak - int(0.2 * self.fs))
            baseline_end = max(0, r_peak - int(0.05 * self.fs))
            baseline = np.mean(ecg_signal[baseline_start:baseline_end])
            
            # ST segment
            st_start = r_peak + st_start_offset
            st_end = r_peak + st_end_offset
            
            if st_end < len(ecg_signal):
                st_segment = ecg_signal[st_start:st_end]
                st_level = np.mean(st_segment)
                
                # ST elevation = ST level - baseline
                st_elevation = st_level - baseline
                st_elevations.append(st_elevation)
        
        features = {
            'st_elevation_mean': np.mean(st_elevations) if st_elevations else 0,
            'st_elevation_std': np.std(st_elevations) if st_elevations else 0,
            'st_elevation_max': np.max(np.abs(st_elevations)) if st_elevations else 0,
        }
        
        print(f"[Feature Extraction] ST segment features:")
        print(f"  Mean ST elevation: {features['st_elevation_mean']:.4f}")
        
        return features
    
    def extract_t_wave_features(self, ecg_signal: np.ndarray, 
                               r_peaks: np.ndarray) -> Dict[str, float]:
        """
        Extract T-wave features
        
        T-wave represents ventricular repolarization
        
        Args:
            ecg_signal: Filtered ECG signal
            r_peaks: R-peak locations
            
        Returns:
            Dictionary of T-wave features
        """
        t_amplitudes = []
        
        # T-wave typically occurs 200-400ms after R-peak
        t_start_offset = int(0.2 * self.fs)
        t_end_offset = int(0.4 * self.fs)
        
        for r_peak in r_peaks:
            t_start = r_peak + t_start_offset
            t_end = r_peak + t_end_offset
            
            if t_end < len(ecg_signal):
                t_segment = ecg_signal[t_start:t_end]
                
                # T-wave amplitude (peak in segment)
                t_amplitude = np.max(np.abs(t_segment))
                t_amplitudes.append(t_amplitude)
        
        features = {
            't_wave_amplitude_mean': np.mean(t_amplitudes) if t_amplitudes else 0,
            't_wave_amplitude_std': np.std(t_amplitudes) if t_amplitudes else 0,
        }
        
        print(f"[Feature Extraction] T-wave features:")
        print(f"  Mean T-wave amplitude: {features['t_wave_amplitude_mean']:.3f}")
        
        return features
    
    def extract_hrv_features(self, r_peaks: np.ndarray) -> Dict[str, float]:
        """
        Extract Heart Rate Variability features
        
        Args:
            r_peaks: R-peak locations
            
        Returns:
            Dictionary of HRV features
        """
        if len(r_peaks) < 2:
            return {}
        
        # RR intervals
        rr_intervals = np.diff(r_peaks) / self.fs * 1000  # in ms
        
        # Time-domain HRV metrics
        mean_rr = np.mean(rr_intervals)
        sdnn = np.std(rr_intervals)
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
        
        # NN50: number of pairs of successive NNs that differ by more than 50 ms
        nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
        pnn50 = (nn50 / len(rr_intervals)) * 100 if len(rr_intervals) > 0 else 0
        
        # Coefficient of variation
        cv = (sdnn / mean_rr) * 100 if mean_rr > 0 else 0
        
        features = {
            'hrv_mean_rr_ms': mean_rr,
            'hrv_sdnn_ms': sdnn,
            'hrv_rmssd_ms': rmssd,
            'hrv_pnn50': pnn50,
            'hrv_cv': cv,
            'mean_hr_bpm': 60000 / mean_rr if mean_rr > 0 else 0,
        }
        
        print(f"[Feature Extraction] HRV features:")
        print(f"  SDNN: {sdnn:.1f} ms, RMSSD: {rmssd:.1f} ms")
        
        return features
    
    def extract_statistical_features(self, ecg_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract statistical features from ECG signal
        
        Args:
            ecg_signal: Input ECG signal
            
        Returns:
            Dictionary of statistical features
        """
        features = {
            'signal_mean': np.mean(ecg_signal),
            'signal_std': np.std(ecg_signal),
            'signal_var': np.var(ecg_signal),
            'signal_skewness': stats.skew(ecg_signal),
            'signal_kurtosis': stats.kurtosis(ecg_signal),
            'signal_energy': np.sum(ecg_signal ** 2),
            'signal_rms': np.sqrt(np.mean(ecg_signal ** 2)),
        }
        
        return features
    
    # =========================================================================
    # Frequency-Domain Features
    # =========================================================================
    
    def extract_spectral_features(self, ecg_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency-domain features using FFT
        
        Lab Exp 8: DCT/DWT features
        
        Args:
            ecg_signal: Input ECG signal
            
        Returns:
            Dictionary of spectral features
        """
        # Compute FFT
        fft_coeffs = np.fft.fft(ecg_signal)
        magnitude = np.abs(fft_coeffs)
        power = magnitude ** 2
        
        # Frequency axis
        N = len(ecg_signal)
        frequencies = np.fft.fftfreq(N, 1/self.fs)
        positive_freq_idx = frequencies >= 0
        
        frequencies = frequencies[positive_freq_idx]
        power = power[positive_freq_idx]
        
        # Spectral features
        total_power = np.sum(power)
        
        # Power in different frequency bands
        vlf_band = (frequencies >= 0.003) & (frequencies < 0.04)  # Very low freq
        lf_band = (frequencies >= 0.04) & (frequencies < 0.15)   # Low freq
        hf_band = (frequencies >= 0.15) & (frequencies < 0.4)    # High freq
        
        vlf_power = np.sum(power[vlf_band])
        lf_power = np.sum(power[lf_band])
        hf_power = np.sum(power[hf_band])
        
        # LF/HF ratio (indicator of autonomic balance)
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
        
        # Spectral entropy
        power_normalized = power / total_power if total_power > 0 else power
        power_normalized = power_normalized[power_normalized > 0]
        spectral_entropy = -np.sum(power_normalized * np.log2(power_normalized))
        
        features = {
            'spectral_total_power': total_power,
            'spectral_vlf_power': vlf_power,
            'spectral_lf_power': lf_power,
            'spectral_hf_power': hf_power,
            'spectral_lf_hf_ratio': lf_hf_ratio,
            'spectral_entropy': spectral_entropy,
        }
        
        print(f"[Feature Extraction] Spectral features:")
        print(f"  LF/HF ratio: {lf_hf_ratio:.2f}")
        
        return features
    
    def extract_wavelet_features(self, ecg_signal: np.ndarray, 
                                wavelet: str = 'db4', 
                                level: int = 5) -> Dict[str, float]:
        """
        Extract wavelet-based features
        
        Lab Exp 8: DWT feature extraction
        
        Args:
            ecg_signal: Input ECG signal
            wavelet: Wavelet type
            level: Decomposition level
            
        Returns:
            Dictionary of wavelet features
        """
        import pywt
        
        # Perform DWT
        coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)
        
        features = {}
        
        # Approximation coefficients
        approx = coeffs[0]
        features['wavelet_approx_energy'] = np.sum(approx ** 2)
        features['wavelet_approx_mean'] = np.mean(approx)
        features['wavelet_approx_std'] = np.std(approx)
        features['wavelet_approx_entropy'] = self._compute_entropy(approx)
        
        # Detail coefficients
        details = coeffs[1:]
        for i, detail in enumerate(details):
            features[f'wavelet_d{i+1}_energy'] = np.sum(detail ** 2)
            features[f'wavelet_d{i+1}_mean'] = np.mean(detail)
            features[f'wavelet_d{i+1}_std'] = np.std(detail)
            features[f'wavelet_d{i+1}_entropy'] = self._compute_entropy(detail)
        
        print(f"[Feature Extraction] Extracted {len(features)} wavelet features")
        
        return features
    
    def _compute_entropy(self, coeffs: np.ndarray) -> float:
        """
        Compute Shannon entropy of coefficients
        
        Args:
            coeffs: Wavelet coefficients
            
        Returns:
            Entropy value
        """
        # Normalize to probability distribution
        abs_coeffs = np.abs(coeffs)
        total = np.sum(abs_coeffs)
        if total == 0:
            return 0
        
        prob = abs_coeffs / total
        prob = prob[prob > 0]
        
        entropy = -np.sum(prob * np.log2(prob))
        
        return entropy
    
    # =========================================================================
    # 2D Scalogram Features (Texture)
    # =========================================================================
    
    def extract_scalogram_texture_features(self, scalogram: np.ndarray) -> Dict[str, float]:
        """
        Extract texture features from scalogram (treated as image)
        
        Lab Exp 8: Texture analysis
        
        Args:
            scalogram: 2D scalogram array
            
        Returns:
            Dictionary of texture features
        """
        # Normalize scalogram to [0, 255]
        normalized = (scalogram - scalogram.min()) / (scalogram.max() - scalogram.min())
        image = (normalized * 255).astype(np.uint8)
        
        # Statistical texture features
        features = {
            'texture_mean': np.mean(image),
            'texture_std': np.std(image),
            'texture_variance': np.var(image),
            'texture_energy': np.sum(image ** 2),
            'texture_entropy': self._compute_entropy(image.ravel()),
        }
        
        # Compute gradient-based features
        grad_x = np.diff(image, axis=1)
        grad_y = np.diff(image, axis=0)
        
        features['texture_grad_x_mean'] = np.mean(np.abs(grad_x))
        features['texture_grad_y_mean'] = np.mean(np.abs(grad_y))
        
        print(f"[Feature Extraction] Extracted scalogram texture features")
        
        return features
    
    # =========================================================================
    # Complete Feature Extraction Pipeline
    # =========================================================================
    
    def extract_all_features(self, ecg_signal: np.ndarray, 
                           r_peaks: np.ndarray,
                           scalogram: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract all features from ECG signal
        
        Args:
            ecg_signal: Filtered ECG signal
            r_peaks: R-peak locations
            scalogram: Optional 2D scalogram
            
        Returns:
            Dictionary with all features
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE FEATURE EXTRACTION")
        print("="*70)
        
        all_features = {}
        
        # Time-domain features
        print("\n[1/6] Extracting QRS features...")
        all_features.update(self.extract_qrs_features(ecg_signal, r_peaks))
        
        print("\n[2/6] Extracting ST segment features...")
        all_features.update(self.extract_st_segment_features(ecg_signal, r_peaks))
        
        print("\n[3/6] Extracting T-wave features...")
        all_features.update(self.extract_t_wave_features(ecg_signal, r_peaks))
        
        print("\n[4/6] Extracting HRV features...")
        all_features.update(self.extract_hrv_features(r_peaks))
        
        # Frequency-domain features
        print("\n[5/6] Extracting spectral features...")
        all_features.update(self.extract_spectral_features(ecg_signal))
        
        # Wavelet features
        print("\n[6/6] Extracting wavelet features...")
        all_features.update(self.extract_wavelet_features(ecg_signal))
        
        # Scalogram features (if provided)
        if scalogram is not None:
            print("\n[Bonus] Extracting scalogram texture features...")
            all_features.update(self.extract_scalogram_texture_features(scalogram))
        
        self.features = all_features
        
        print(f"\n[Feature Extraction] Total features extracted: {len(all_features)}")
        print("="*70)
        
        return all_features


class ECGClassifier:
    """
    Simple ECG classifier for normal vs. abnormal detection
    Uses threshold-based rules and SVM
    """
    
    def __init__(self):
        """Initialize classifier"""
        self.scaler = StandardScaler()
        self.svm_model = None
        self.feature_names = None
        
    def classify_threshold_based(self, features: Dict[str, float]) -> Tuple[str, Dict[str, bool]]:
        """
        Classify using simple threshold-based rules
        
        Rules for abnormal detection:
        - Abnormal heart rate (<60 or >100 BPM)
        - Wide QRS (>120 ms)
        - ST elevation/depression (>0.1)
        - Low HRV (SDNN < 50 ms)
        
        Args:
            features: Feature dictionary
            
        Returns:
            Tuple of (classification, rule_triggers)
        """
        rule_triggers = {}
        
        # Rule 1: Heart rate
        hr = features.get('mean_hr_bpm', 75)
        rule_triggers['abnormal_hr'] = (hr < 60) or (hr > 100)
        
        # Rule 2: Wide QRS
        qrs_width = features.get('qrs_width_mean_ms', 80)
        rule_triggers['wide_qrs'] = qrs_width > 120
        
        # Rule 3: ST elevation/depression
        st_elevation = features.get('st_elevation_max', 0)
        rule_triggers['st_abnormal'] = abs(st_elevation) > 0.1
        
        # Rule 4: Low HRV
        sdnn = features.get('hrv_sdnn_ms', 50)
        rule_triggers['low_hrv'] = sdnn < 50
        
        # Classification: abnormal if any rule triggers
        is_abnormal = any(rule_triggers.values())
        classification = 'ABNORMAL' if is_abnormal else 'NORMAL'
        
        print(f"\n[Threshold Classification] Result: {classification}")
        for rule, triggered in rule_triggers.items():
            if triggered:
                print(f"  ⚠ {rule} triggered")
        
        return classification, rule_triggers
    
    def train_svm_classifier(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train SVM classifier
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (0=normal, 1=abnormal)
        """
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train SVM with RBF kernel
        self.svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        self.svm_model.fit(X_train_scaled, y_train)
        
        print(f"[SVM Classifier] Trained on {len(X_train)} samples")
        
    def predict_svm(self, features: np.ndarray) -> Tuple[int, float]:
        """
        Predict using trained SVM
        
        Args:
            features: Feature vector
            
        Returns:
            Tuple of (prediction, confidence)
        """
        if self.svm_model is None:
            raise ValueError("SVM model not trained")
        
        # Reshape if single sample
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Standardize
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.svm_model.predict(features_scaled)[0]
        probabilities = self.svm_model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        return prediction, confidence


def demo_feature_extraction():
    """
    Demonstrate feature extraction and classification
    """
    print("="*70)
    print("FEATURE EXTRACTION AND CLASSIFICATION DEMONSTRATION")
    print("="*70)
    
    # Initialize
    fs = 360
    extractor = ECGFeatureExtractor(sampling_rate=fs)
    classifier = ECGClassifier()
    
    # Generate synthetic ECG
    duration = 10
    t = np.arange(0, duration, 1/fs)
    
    from scipy import signal
    ecg = np.zeros_like(t)
    for beat_time in np.arange(0.5, duration, 0.8):
        beat_idx = int(beat_time * fs)
        if beat_idx < len(ecg) - 100:
            beat = gaussian(100, 10)
            ecg[beat_idx:beat_idx+100] += beat
    
    # Simulate R-peaks
    r_peaks = np.array([int(t * fs) for t in np.arange(0.5, duration, 0.8)])
    
    # Extract features
    features = extractor.extract_all_features(ecg, r_peaks)
    
    # Classify
    print("\n" + "="*70)
    print("CLASSIFICATION")
    print("="*70)
    classification, rules = classifier.classify_threshold_based(features)
    
    print(f"\n[Demo] Feature extraction and classification completed!")
    print("="*70)
    
    return extractor, classifier, features


if __name__ == "__main__":
    # Run demonstration
    extractor, classifier, features = demo_feature_extraction()
    
    print("\n[Demo] Sample features:")
    for key, value in list(features.items())[:10]:
        print(f"  {key}: {value:.4f}")
