"""
ECG Signal Preprocessing Module
Digital Signal Processing Mini-Project

Implements:
- Unit 1: Discrete-Time Signals & Systems
- Lab Exp 1: Sampling & Reconstruction
- Lab Exp 2: Discrete Correlation & Convolution
- Unit 3: DSP Algorithms (Overlap-Add/Save)
"""

import numpy as np
from scipy import signal, interpolate
from scipy.signal import butter, filtfilt, find_peaks
from scipy.signal.windows import gaussian
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


class ECGPreprocessor:
    """
    ECG Signal Preprocessing Pipeline
    Covers: Sampling, filtering, noise removal, R-peak detection
    """
    
    def __init__(self, sampling_rate: float = 360.0):
        """
        Initialize ECG preprocessor
        
        Args:
            sampling_rate: Original sampling rate in Hz (MIT-BIH default: 360 Hz)
        """
        self.fs = sampling_rate
        self.filtered_signal = None
        self.r_peaks = None
        
    # =========================================================================
    # Unit 1.1 & Lab Exp 1: Sampling & Reconstruction
    # =========================================================================
    
    def downsample_and_reconstruct(self, ecg_signal: np.ndarray, 
                                   downsample_factor: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Demonstrate sampling theorem and signal reconstruction
        
        Unit 1.1: Sampling theorem - signal can be reconstructed if fs > 2*fmax
        Lab Exp 1: Sampling & Reconstruction
        
        Args:
            ecg_signal: Original ECG signal
            downsample_factor: Downsampling factor (default: 4)
            
        Returns:
            Tuple of (downsampled_signal, reconstructed_signal, time_axis)
        """
        # Original time axis
        t_original = np.arange(len(ecg_signal)) / self.fs
        
        # Downsample: take every nth sample
        downsampled = ecg_signal[::downsample_factor]
        t_downsampled = t_original[::downsample_factor]
        
        # Reconstruction using cubic spline interpolation
        # This approximates sinc interpolation for bandlimited signals
        interpolator = interpolate.interp1d(t_downsampled, downsampled, 
                                           kind='cubic', fill_value='extrapolate')
        reconstructed = interpolator(t_original)
        
        print(f"[Lab Exp 1] Original samples: {len(ecg_signal)}")
        print(f"[Lab Exp 1] Downsampled to: {len(downsampled)} samples")
        print(f"[Lab Exp 1] Reconstructed to: {len(reconstructed)} samples")
        
        return downsampled, reconstructed, t_original
    
    # =========================================================================
    # Unit 1.2: Linear Convolution for Filtering
    # =========================================================================
    
    def design_bandpass_filter(self, lowcut: float = 0.5, highcut: float = 40.0, 
                               order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Design Butterworth bandpass FIR filter
        
        Unit 1.2: Filter design and impulse response
        Removes baseline wander (<0.5 Hz) and high-frequency noise (>40 Hz)
        
        Args:
            lowcut: Low cutoff frequency in Hz
            highcut: High cutoff frequency in Hz
            order: Filter order
            
        Returns:
            Tuple of (b_coefficients, a_coefficients)
        """
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Design Butterworth bandpass filter
        b, a = butter(order, [low, high], btype='band')
        
        print(f"[Unit 1.2] Designed bandpass filter: {lowcut}-{highcut} Hz")
        print(f"[Unit 1.2] Filter order: {order}")
        
        return b, a
    
    def apply_convolution_filter(self, ecg_signal: np.ndarray, 
                                 filter_coeffs: np.ndarray) -> np.ndarray:
        """
        Apply filtering using linear convolution
        
        Unit 1.2: Linear Convolution (1D)
        y[n] = sum(h[k] * x[n-k]) - convolution of signal with filter impulse response
        
        Args:
            ecg_signal: Input ECG signal
            filter_coeffs: Filter impulse response (FIR coefficients)
            
        Returns:
            Filtered signal
        """
        # Linear convolution: y[n] = x[n] * h[n]
        filtered = np.convolve(ecg_signal, filter_coeffs, mode='same')
        
        print(f"[Unit 1.2] Applied linear convolution filtering")
        print(f"[Unit 1.2] Input length: {len(ecg_signal)}, Output length: {len(filtered)}")
        
        return filtered
    
    # =========================================================================
    # Unit 3: Overlap-Add Method for Long Sequences
    # =========================================================================
    
    def overlap_add_filter(self, ecg_signal: np.ndarray, 
                          filter_coeffs: np.ndarray, 
                          block_size: int = 512) -> np.ndarray:
        """
        Implement Overlap-Add method for efficient FIR filtering via FFT
        
        Unit 3: DSP Algorithms - Overlap-Add method for FIR filtering
        Efficient for long sequences using FFT-based convolution
        
        Args:
            ecg_signal: Input ECG signal
            filter_coeffs: FIR filter coefficients (impulse response)
            block_size: Block size for processing
            
        Returns:
            Filtered signal
        """
        M = len(filter_coeffs)  # Filter length
        L = block_size  # Block length
        N = L + M - 1  # FFT size (for linear convolution)
        
        # Zero-pad filter to FFT size
        H = np.fft.fft(filter_coeffs, N)
        
        # Initialize output
        num_blocks = int(np.ceil(len(ecg_signal) / L))
        output = np.zeros(len(ecg_signal) + M - 1)
        
        print(f"[Unit 3] Overlap-Add: Processing {num_blocks} blocks of size {L}")
        
        for i in range(num_blocks):
            # Extract block
            start_idx = i * L
            end_idx = min(start_idx + L, len(ecg_signal))
            block = ecg_signal[start_idx:end_idx]
            
            # Zero-pad block to FFT size
            block_padded = np.zeros(N)
            block_padded[:len(block)] = block
            
            # FFT-based convolution: IFFT(FFT(x) * FFT(h))
            X = np.fft.fft(block_padded)
            Y = X * H
            y = np.fft.ifft(Y).real
            
            # Overlap-add: accumulate result
            output[start_idx:start_idx + N] += y
        
        # Trim to original signal length
        filtered = output[:len(ecg_signal)]
        
        print(f"[Unit 3] Overlap-Add filtering completed")
        
        return filtered
    
    def filter_ecg(self, ecg_signal: np.ndarray, 
                   method: str = 'filtfilt',
                   lowcut: float = 0.5, 
                   highcut: float = 40.0) -> np.ndarray:
        """
        Apply bandpass filtering to ECG signal
        
        Args:
            ecg_signal: Raw ECG signal
            method: Filtering method ('filtfilt', 'convolve', 'overlap_add')
            lowcut: Low cutoff frequency
            highcut: High cutoff frequency
            
        Returns:
            Filtered ECG signal
        """
        b, a = self.design_bandpass_filter(lowcut, highcut)
        
        if method == 'filtfilt':
            # Zero-phase filtering (most common for ECG)
            filtered = filtfilt(b, a, ecg_signal)
        elif method == 'convolve':
            # Using linear convolution with FIR approximation
            # Convert IIR to FIR impulse response
            impulse = np.zeros(1000)
            impulse[0] = 1
            h = filtfilt(b, a, impulse)
            filtered = self.apply_convolution_filter(ecg_signal, h[:100])
        elif method == 'overlap_add':
            # Overlap-add method (Unit 3)
            impulse = np.zeros(1000)
            impulse[0] = 1
            h = filtfilt(b, a, impulse)
            filtered = self.overlap_add_filter(ecg_signal, h[:100])
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.filtered_signal = filtered
        return filtered
    
    # =========================================================================
    # Lab Exp 2: Cross-Correlation for R-Peak Detection
    # =========================================================================
    
    def detect_r_peaks_correlation(self, ecg_signal: np.ndarray, 
                                   distance: int = 200) -> np.ndarray:
        """
        Detect R-peaks using cross-correlation with QRS template
        
        Lab Exp 2: Discrete Correlation
        Cross-correlation measures similarity between signal and template
        
        Args:
            ecg_signal: Filtered ECG signal
            distance: Minimum distance between peaks (samples)
            
        Returns:
            Array of R-peak indices
        """
        # Create synthetic QRS template (simplified)
        template_duration = int(0.1 * self.fs)  # 100ms QRS complex
        t = np.linspace(0, 1, template_duration)
        qrs_template = np.exp(-((t - 0.5) ** 2) / 0.02) * np.sin(2 * np.pi * t * 5)
        qrs_template = qrs_template / np.max(np.abs(qrs_template))
        
        # Compute cross-correlation
        # Lab Exp 2: r_xy[n] = sum(x[m] * y[m+n])
        correlation = np.correlate(ecg_signal, qrs_template, mode='same')
        
        # Find peaks in correlation signal
        peaks, properties = find_peaks(correlation, distance=distance, 
                                       height=np.max(correlation) * 0.3)
        
        self.r_peaks = peaks
        
        print(f"[Lab Exp 2] Detected {len(peaks)} R-peaks using cross-correlation")
        
        return peaks
    
    def detect_r_peaks_simple(self, ecg_signal: np.ndarray, 
                             distance: Optional[int] = None) -> np.ndarray:
        """
        Simple R-peak detection using adaptive thresholding
        
        Args:
            ecg_signal: Filtered ECG signal
            distance: Minimum distance between peaks (default: 0.6*fs)
            
        Returns:
            Array of R-peak indices
        """
        if distance is None:
            distance = int(0.6 * self.fs)  # Minimum 600ms between beats
        
        # Compute derivative to emphasize QRS slope
        derivative = np.diff(ecg_signal)
        derivative = np.abs(derivative)
        
        # Adaptive threshold
        threshold = np.mean(derivative) + 0.5 * np.std(derivative)
        
        # Find peaks
        peaks, _ = find_peaks(ecg_signal, distance=distance, height=threshold * 0.5)
        
        self.r_peaks = peaks
        
        print(f"[Simple Detection] Detected {len(peaks)} R-peaks")
        
        return peaks
    
    # =========================================================================
    # Unit 1.3: System Response Modeling
    # =========================================================================
    
    def model_ecg_system(self, impulse_duration: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Model ECG as output of LTI system with QRS impulse response
        
        Unit 1.3: System output via convolution
        y[n] = x[n] * h[n] where h[n] is the system impulse response
        
        Args:
            impulse_duration: Duration of QRS impulse response in seconds
            
        Returns:
            Tuple of (impulse_response, synthetic_ecg)
        """
        # Create QRS-like impulse response
        n_samples = int(impulse_duration * self.fs)
        t = np.linspace(0, 1, n_samples)
        
        # Model as combination of Gaussians (P, QRS, T waves)
        h = (0.3 * np.exp(-((t - 0.2) ** 2) / 0.005) +  # P wave
             1.0 * np.exp(-((t - 0.5) ** 2) / 0.002) -  # QRS (positive)
             0.5 * np.exp(-((t - 0.48) ** 2) / 0.001) +  # QRS (negative)
             0.4 * np.exp(-((t - 0.7) ** 2) / 0.008))    # T wave
        
        # Create impulse train input (heartbeat events)
        input_signal = np.zeros(int(5 * self.fs))
        heartbeat_times = [0.8, 1.6, 2.4, 3.2, 4.0]
        for beat_time in heartbeat_times:
            idx = int(beat_time * self.fs)
            if idx < len(input_signal):
                input_signal[idx] = 1.0
        
        # System output via linear convolution
        # Unit 1.3: Output = Input convolved with impulse response
        synthetic_ecg = np.convolve(input_signal, h, mode='same')
        
        print(f"[Unit 1.3] Modeled ECG as LTI system output")
        print(f"[Unit 1.3] Impulse response length: {len(h)} samples")
        
        return h, synthetic_ecg
    
    # =========================================================================
    # Heart Rate Variability (HRV) Analysis
    # =========================================================================
    
    def calculate_hrv(self, r_peaks: Optional[np.ndarray] = None) -> dict:
        """
        Calculate Heart Rate Variability metrics
        
        Args:
            r_peaks: R-peak indices (uses self.r_peaks if None)
            
        Returns:
            Dictionary with HRV metrics (SDNN, RMSSD, mean_hr, etc.)
        """
        if r_peaks is None:
            r_peaks = self.r_peaks
        
        if r_peaks is None or len(r_peaks) < 2:
            raise ValueError("Need at least 2 R-peaks for HRV calculation")
        
        # Calculate RR intervals (in milliseconds)
        rr_intervals = np.diff(r_peaks) / self.fs * 1000  # Convert to ms
        
        # Time-domain HRV metrics
        sdnn = np.std(rr_intervals)  # Standard deviation of NN intervals
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))  # Root mean square of successive differences
        mean_rr = np.mean(rr_intervals)
        mean_hr = 60000 / mean_rr  # Heart rate in BPM
        
        hrv_metrics = {
            'mean_rr_ms': mean_rr,
            'mean_hr_bpm': mean_hr,
            'sdnn_ms': sdnn,
            'rmssd_ms': rmssd,
            'num_beats': len(r_peaks)
        }
        
        print(f"\n[HRV Analysis]")
        print(f"  Mean Heart Rate: {mean_hr:.1f} BPM")
        print(f"  Mean RR Interval: {mean_rr:.1f} ms")
        print(f"  SDNN: {sdnn:.1f} ms")
        print(f"  RMSSD: {rmssd:.1f} ms")
        
        return hrv_metrics
    
    # =========================================================================
    # Signal Standardization
    # =========================================================================
    
    def normalize_signal(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Normalize ECG signal amplitude to [-1, 1]
        
        Args:
            ecg_signal: Input ECG signal
            
        Returns:
            Normalized signal
        """
        # Remove DC offset
        signal_centered = ecg_signal - np.mean(ecg_signal)
        
        # Normalize to [-1, 1]
        max_abs = np.max(np.abs(signal_centered))
        if max_abs > 0:
            normalized = signal_centered / max_abs
        else:
            normalized = signal_centered
        
        print(f"[Normalization] Signal normalized to range [{np.min(normalized):.2f}, {np.max(normalized):.2f}]")
        
        return normalized
    
    # =========================================================================
    # Visualization
    # =========================================================================
    
    def plot_preprocessing_results(self, original: np.ndarray, 
                                   filtered: np.ndarray,
                                   r_peaks: Optional[np.ndarray] = None,
                                   duration: float = 5.0):
        """
        Plot preprocessing results
        
        Args:
            original: Original ECG signal
            filtered: Filtered ECG signal
            r_peaks: R-peak locations
            duration: Duration to plot (seconds)
        """
        samples = int(duration * self.fs)
        t = np.arange(samples) / self.fs
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        
        # Original signal
        axes[0].plot(t, original[:samples], 'b-', linewidth=0.8, label='Original')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Original ECG Signal')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Filtered signal with R-peaks
        axes[1].plot(t, filtered[:samples], 'g-', linewidth=0.8, label='Filtered')
        if r_peaks is not None:
            r_peaks_in_range = r_peaks[r_peaks < samples]
            axes[1].plot(r_peaks_in_range / self.fs, 
                        filtered[r_peaks_in_range], 
                        'ro', markersize=8, label='R-peaks')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_title('Filtered ECG Signal with R-peaks')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        return fig


def demo_preprocessing():
    """
    Demonstrate ECG preprocessing with synthetic data
    """
    print("="*70)
    print("ECG PREPROCESSING DEMONSTRATION")
    print("="*70)
    
    # Initialize preprocessor
    fs = 360  # Hz
    preprocessor = ECGPreprocessor(sampling_rate=fs)
    
    # Generate synthetic ECG signal
    duration = 10  # seconds
    t = np.arange(0, duration, 1/fs)
    
    # Create synthetic ECG with multiple heartbeats
    ecg = np.zeros_like(t)
    heart_rate = 75  # BPM
    beat_interval = 60 / heart_rate
    
    for beat_time in np.arange(0.5, duration, beat_interval):
        beat_idx = int(beat_time * fs)
        if beat_idx < len(ecg) - 100:
            # Simplified PQRST complex
            beat = gaussian(100, 10)
            ecg[beat_idx:beat_idx+100] += beat
    
    # Add noise
    noise = 0.1 * np.random.randn(len(ecg))
    baseline_wander = 0.2 * np.sin(2 * np.pi * 0.3 * t)
    noisy_ecg = ecg + noise + baseline_wander
    
    print("\n[Demo] Generated synthetic ECG signal")
    print(f"[Demo] Duration: {duration} seconds, Sampling rate: {fs} Hz")
    
    # Test sampling and reconstruction
    print("\n" + "="*70)
    print("LAB EXP 1: SAMPLING & RECONSTRUCTION")
    print("="*70)
    downsampled, reconstructed, t_axis = preprocessor.downsample_and_reconstruct(noisy_ecg)
    
    # Test filtering
    print("\n" + "="*70)
    print("UNIT 1.2 & UNIT 3: FILTERING")
    print("="*70)
    filtered = preprocessor.filter_ecg(noisy_ecg, method='filtfilt')
    
    # Test R-peak detection
    print("\n" + "="*70)
    print("LAB EXP 2: R-PEAK DETECTION")
    print("="*70)
    r_peaks = preprocessor.detect_r_peaks_correlation(filtered)
    
    # Calculate HRV
    print("\n" + "="*70)
    print("HEART RATE VARIABILITY")
    print("="*70)
    hrv_metrics = preprocessor.calculate_hrv(r_peaks)
    
    # Normalize
    normalized = preprocessor.normalize_signal(filtered)
    
    print("\n[Demo] Preprocessing demonstration completed successfully!")
    print("="*70)
    
    return preprocessor, noisy_ecg, filtered, r_peaks


if __name__ == "__main__":
    # Run demonstration
    preprocessor, original, filtered, r_peaks = demo_preprocessing()
    
    # Visualize results
    plt.style.use('seaborn-v0_8-darkgrid')
    preprocessor.plot_preprocessing_results(original, filtered, r_peaks, duration=5.0)
    plt.savefig('/tmp/ecg_preprocessing_demo.png', dpi=150, bbox_inches='tight')
    print("\n[Demo] Saved visualization to /tmp/ecg_preprocessing_demo.png")
    plt.show()
