"""
ECG Frequency Domain Analysis Module
Digital Signal Processing Mini-Project

Implements:
- Unit 2: DFT/FFT
- Lab Exp 3: DFT Implementation
- Lab Exp 4: FFT Implementation
- Lab Exp 8: DWT on Signals
- Unit 2.5: Wavelet Transform
"""

import numpy as np
import pywt
import matplotlib.pyplot as plt
from typing import Tuple, List
from scipy.signal.windows import gaussian
import time


class FrequencyAnalyzer:
    """
    Frequency domain analysis for ECG signals
    Includes DFT, FFT, and Wavelet transforms
    """
    
    def __init__(self, sampling_rate: float = 360.0):
        """
        Initialize frequency analyzer
        
        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.fs = sampling_rate
    
    # =========================================================================
    # Lab Exp 3: DFT Implementation
    # =========================================================================
    
    def compute_dft(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute Discrete Fourier Transform from first principles
        
        Lab Exp 3: DFT Implementation
        Unit 2.1: DTFT and DFT
        
        DFT Formula: X[k] = sum(x[n] * exp(-j*2*pi*k*n/N)) for n=0 to N-1
        
        Args:
            signal: Input signal
            
        Returns:
            DFT coefficients (complex)
        """
        N = len(signal)
        X = np.zeros(N, dtype=complex)
        
        print(f"[Lab Exp 3] Computing DFT for signal of length {N}")
        
        # Direct DFT computation: O(N^2) complexity
        for k in range(N):
            for n in range(N):
                X[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
        
        print(f"[Lab Exp 3] DFT computation completed")
        
        return X
    
    # =========================================================================
    # Unit 2.3 & Lab Exp 4: Radix-2 DIT-FFT Implementation
    # =========================================================================
    
    def fft_radix2_dit(self, signal: np.ndarray) -> np.ndarray:
        """
        Implement Radix-2 Decimation-In-Time FFT algorithm
        
        Unit 2.3: Radix-2 DIT-FFT
        Lab Exp 4: FFT Implementation
        
        Cooley-Tukey FFT algorithm with O(N log N) complexity
        Divides DFT into even and odd indexed samples recursively
        
        Args:
            signal: Input signal (length must be power of 2)
            
        Returns:
            FFT coefficients (complex)
        """
        N = len(signal)
        
        # Base case: DFT of single point
        if N <= 1:
            return signal
        
        # Check if N is power of 2
        if N & (N - 1) != 0:
            # Pad to next power of 2
            next_pow2 = 2 ** int(np.ceil(np.log2(N)))
            signal = np.pad(signal, (0, next_pow2 - N), mode='constant')
            N = next_pow2
        
        # Unit 2.3: Decimation-In-Time (DIT)
        # Divide: split into even and odd indexed samples
        even = self.fft_radix2_dit(signal[0::2])
        odd = self.fft_radix2_dit(signal[1::2])
        
        # Conquer: combine using twiddle factors
        # W_N^k = exp(-2*pi*j*k/N) - twiddle factor
        T = np.exp(-2j * np.pi * np.arange(N // 2) / N)
        
        # Butterfly computation
        X = np.zeros(N, dtype=complex)
        X[:N//2] = even + T * odd
        X[N//2:] = even - T * odd
        
        return X
    
    def compare_dft_fft(self, signal: np.ndarray, max_length: int = 512) -> dict:
        """
        Compare DFT and FFT computation times
        
        Unit 2.4: FFT vs DFT performance comparison
        
        Args:
            signal: Input signal
            max_length: Maximum length for DFT (for practical timing)
            
        Returns:
            Dictionary with timing results
        """
        print("\n" + "="*70)
        print("UNIT 2.4: DFT vs FFT PERFORMANCE COMPARISON")
        print("="*70)
        
        # Use shorter signal for DFT (O(N^2) is slow)
        signal_dft = signal[:min(len(signal), max_length)]
        
        # Time DFT
        start_time = time.time()
        X_dft = self.compute_dft(signal_dft)
        dft_time = time.time() - start_time
        
        # Time custom FFT
        start_time = time.time()
        X_fft_custom = self.fft_radix2_dit(signal_dft)
        fft_custom_time = time.time() - start_time
        
        # Time NumPy FFT (optimized C implementation)
        start_time = time.time()
        X_fft_numpy = np.fft.fft(signal_dft)
        fft_numpy_time = time.time() - start_time
        
        # Verify correctness (compare magnitudes)
        error_custom = np.max(np.abs(np.abs(X_dft) - np.abs(X_fft_custom)))
        error_numpy = np.max(np.abs(np.abs(X_dft) - np.abs(X_fft_numpy)))
        
        results = {
            'signal_length': len(signal_dft),
            'dft_time': dft_time,
            'fft_custom_time': fft_custom_time,
            'fft_numpy_time': fft_numpy_time,
            'speedup_custom': dft_time / fft_custom_time if fft_custom_time > 0 else 0,
            'speedup_numpy': dft_time / fft_numpy_time if fft_numpy_time > 0 else 0,
            'error_custom': error_custom,
            'error_numpy': error_numpy
        }
        
        print(f"\n[Performance] Signal length: {len(signal_dft)}")
        print(f"[Performance] DFT time: {dft_time:.4f} seconds")
        print(f"[Performance] Custom FFT time: {fft_custom_time:.4f} seconds")
        print(f"[Performance] NumPy FFT time: {fft_numpy_time:.4f} seconds")
        print(f"[Performance] Speedup (Custom FFT vs DFT): {results['speedup_custom']:.1f}x")
        print(f"[Performance] Speedup (NumPy FFT vs DFT): {results['speedup_numpy']:.1f}x")
        print(f"[Performance] Max error (Custom): {error_custom:.2e}")
        print(f"[Performance] Max error (NumPy): {error_numpy:.2e}")
        
        return results
    
    # =========================================================================
    # Unit 2.2: DFT Properties and Spectral Analysis
    # =========================================================================
    
    def compute_magnitude_spectrum(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute magnitude spectrum of signal
        
        Unit 2.2: DFT properties
        |X[k]| gives magnitude at frequency k*fs/N
        
        Args:
            signal: Input signal
            
        Returns:
            Tuple of (frequencies, magnitudes)
        """
        # Compute FFT using NumPy for efficiency
        N = len(signal)
        X = np.fft.fft(signal)
        
        # Magnitude spectrum
        magnitude = np.abs(X)
        
        # Frequency axis
        frequencies = np.fft.fftfreq(N, 1/self.fs)
        
        # Only return positive frequencies (symmetric for real signals)
        positive_freq_idx = frequencies >= 0
        
        return frequencies[positive_freq_idx], magnitude[positive_freq_idx]
    
    def compute_power_spectrum(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectral density
        
        Unit 2.2: Parseval's theorem
        Total power in time domain = Total power in frequency domain
        
        Args:
            signal: Input signal
            
        Returns:
            Tuple of (frequencies, power)
        """
        frequencies, magnitude = self.compute_magnitude_spectrum(signal)
        
        # Power spectral density: |X[k]|^2 / N
        N = len(signal)
        power = (magnitude ** 2) / N
        
        print(f"[Unit 2.2] Computed power spectrum")
        print(f"[Unit 2.2] Parseval's theorem: Total time-domain power = {np.sum(signal**2):.2f}")
        print(f"[Unit 2.2] Total frequency-domain power = {np.sum(power):.2f}")
        
        return frequencies, power
    
    def identify_frequency_components(self, signal: np.ndarray, 
                                     num_peaks: int = 5) -> List[Tuple[float, float]]:
        """
        Identify dominant frequency components in ECG signal
        
        Args:
            signal: Input ECG signal
            num_peaks: Number of peaks to identify
            
        Returns:
            List of (frequency, magnitude) tuples
        """
        frequencies, magnitude = self.compute_magnitude_spectrum(signal)
        
        # Find peaks in magnitude spectrum
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(magnitude, height=np.max(magnitude) * 0.1)
        
        # Sort by magnitude
        sorted_idx = np.argsort(properties['peak_heights'])[::-1]
        top_peaks = peaks[sorted_idx[:num_peaks]]
        
        components = [(frequencies[p], magnitude[p]) for p in top_peaks]
        
        print(f"\n[Frequency Analysis] Top {num_peaks} frequency components:")
        for freq, mag in components:
            print(f"  {freq:.2f} Hz: magnitude {mag:.1f}")
        
        return components
    
    # =========================================================================
    # Unit 2.5 & Lab Exp 8: Wavelet Transform
    # =========================================================================
    
    def apply_dwt(self, signal: np.ndarray, wavelet: str = 'db4', 
                  level: int = 5) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Apply Discrete Wavelet Transform for multi-resolution analysis
        
        Unit 2.5: Wavelet Transform basics
        Lab Exp 8: DWT on Signals
        
        DWT decomposes signal into approximation (low-freq) and details (high-freq)
        
        Args:
            signal: Input ECG signal
            wavelet: Wavelet type (default: Daubechies 4)
            level: Decomposition level
            
        Returns:
            Tuple of (approximation_coeffs, detail_coeffs_list)
        """
        print(f"\n[Lab Exp 8] Applying DWT with wavelet '{wavelet}', level {level}")
        
        # Perform multi-level DWT
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        
        # First element is approximation, rest are details
        approximation = coeffs[0]
        details = coeffs[1:]
        
        print(f"[Lab Exp 8] Approximation coefficients: {len(approximation)}")
        for i, detail in enumerate(details):
            print(f"[Lab Exp 8] Detail level {i+1}: {len(detail)} coefficients")
        
        return approximation, details
    
    def apply_cwt(self, signal: np.ndarray, wavelet: str = 'morl', 
                  scales: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Continuous Wavelet Transform for time-frequency analysis
        
        Unit 2.5: Wavelet Transform
        CWT provides time-frequency localization (like spectrogram but better)
        
        Args:
            signal: Input ECG signal
            wavelet: Wavelet type (default: Morlet)
            scales: Array of scales (related to frequency)
            
        Returns:
            Tuple of (coefficients, frequencies)
        """
        if scales is None:
            # Default scales corresponding to meaningful frequencies
            scales = np.arange(1, 128)
        
        print(f"\n[Unit 2.5] Applying CWT with wavelet '{wavelet}'")
        
        # Compute CWT
        coefficients, frequencies = pywt.cwt(signal, scales, wavelet, 1/self.fs)
        
        print(f"[Unit 2.5] CWT shape: {coefficients.shape}")
        print(f"[Unit 2.5] Frequency range: {frequencies.min():.2f} - {frequencies.max():.2f} Hz")
        
        return coefficients, frequencies
    
    def denoise_with_dwt(self, signal: np.ndarray, wavelet: str = 'db4', 
                        level: int = 5, threshold_scale: float = 0.5) -> np.ndarray:
        """
        Denoise signal using wavelet thresholding
        
        Lab Exp 8: DWT for noise suppression
        
        Args:
            signal: Noisy input signal
            wavelet: Wavelet type
            level: Decomposition level
            threshold_scale: Threshold multiplier
            
        Returns:
            Denoised signal
        """
        # Decompose
        approximation, details = self.apply_dwt(signal, wavelet, level)
        
        # Threshold detail coefficients (soft thresholding)
        details_thresholded = []
        for detail in details:
            # Universal threshold: sigma * sqrt(2 * log(N))
            sigma = np.median(np.abs(detail)) / 0.6745
            threshold = threshold_scale * sigma * np.sqrt(2 * np.log(len(signal)))
            
            # Soft thresholding
            detail_thresh = pywt.threshold(detail, threshold, mode='soft')
            details_thresholded.append(detail_thresh)
        
        # Reconstruct
        coeffs_thresh = [approximation] + details_thresholded
        denoised = pywt.waverec(coeffs_thresh, wavelet)
        
        # Handle length mismatch due to padding
        denoised = denoised[:len(signal)]
        
        print(f"[Lab Exp 8] Denoised signal using wavelet thresholding")
        
        return denoised
    
    def extract_wavelet_features(self, signal: np.ndarray, wavelet: str = 'db4', 
                                level: int = 5) -> dict:
        """
        Extract features from wavelet coefficients
        
        Lab Exp 8: Feature extraction using DWT
        
        Args:
            signal: Input signal
            wavelet: Wavelet type
            level: Decomposition level
            
        Returns:
            Dictionary of wavelet-based features
        """
        approximation, details = self.apply_dwt(signal, wavelet, level)
        
        features = {
            'approx_energy': np.sum(approximation ** 2),
            'approx_mean': np.mean(approximation),
            'approx_std': np.std(approximation),
        }
        
        # Detail features for each level
        for i, detail in enumerate(details):
            features[f'detail{i+1}_energy'] = np.sum(detail ** 2)
            features[f'detail{i+1}_mean'] = np.mean(detail)
            features[f'detail{i+1}_std'] = np.std(detail)
        
        print(f"\n[Lab Exp 8] Extracted {len(features)} wavelet-based features")
        
        return features
    
    # =========================================================================
    # Visualization
    # =========================================================================
    
    def plot_frequency_analysis(self, signal: np.ndarray, 
                               duration: float = 5.0,
                               signal_label: str = 'ECG Signal'):
        """
        Comprehensive frequency domain visualization
        
        Args:
            signal: Input signal
            duration: Duration to display (seconds)
            signal_label: Label for signal
        """
        samples = int(duration * self.fs)
        signal_segment = signal[:samples]
        t = np.arange(samples) / self.fs
        
        # Compute spectra
        frequencies, magnitude = self.compute_magnitude_spectrum(signal_segment)
        frequencies_power, power = self.compute_power_spectrum(signal_segment)
        
        # Create figure
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Time domain
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(t, signal_segment, 'b-', linewidth=0.8)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'{signal_label} - Time Domain')
        ax1.grid(True, alpha=0.3)
        
        # Magnitude spectrum
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(frequencies, magnitude, 'r-', linewidth=0.8)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Magnitude Spectrum (DFT)')
        ax2.set_xlim([0, 50])
        ax2.grid(True, alpha=0.3)
        
        # Power spectrum
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.semilogy(frequencies_power, power, 'g-', linewidth=0.8)
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Power (log scale)')
        ax3.set_title('Power Spectrum')
        ax3.set_xlim([0, 50])
        ax3.grid(True, alpha=0.3)
        
        # Spectrogram
        ax4 = fig.add_subplot(gs[2, 0])
        from scipy import signal as sp_signal
        f, t_spec, Sxx = sp_signal.spectrogram(signal_segment, self.fs, 
                                                nperseg=256, noverlap=128)
        im = ax4.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
        ax4.set_ylabel('Frequency (Hz)')
        ax4.set_xlabel('Time (s)')
        ax4.set_title('Spectrogram (FFT-based)')
        ax4.set_ylim([0, 50])
        plt.colorbar(im, ax=ax4, label='Power (dB)')
        
        # Scalogram (CWT)
        ax5 = fig.add_subplot(gs[2, 1])
        scales = np.arange(1, 128)
        coefficients, freqs = self.apply_cwt(signal_segment, scales=scales)
        im = ax5.pcolormesh(t, freqs, np.abs(coefficients), shading='gouraud', cmap='viridis')
        ax5.set_ylabel('Frequency (Hz)')
        ax5.set_xlabel('Time (s)')
        ax5.set_title('Scalogram (CWT - Wavelet Transform)')
        ax5.set_ylim([0, 50])
        plt.colorbar(im, ax=ax5, label='Magnitude')
        
        return fig
    
    def plot_wavelet_decomposition(self, signal: np.ndarray, 
                                   wavelet: str = 'db4', 
                                   level: int = 5,
                                   duration: float = 5.0):
        """
        Plot wavelet decomposition results
        
        Args:
            signal: Input signal
            wavelet: Wavelet type
            level: Decomposition level
            duration: Duration to display
        """
        samples = int(duration * self.fs)
        signal_segment = signal[:samples]
        
        # Perform DWT
        approximation, details = self.apply_dwt(signal_segment, wavelet, level)
        
        # Create figure
        fig, axes = plt.subplots(level + 2, 1, figsize=(12, 2*(level+2)))
        
        # Original signal
        t = np.arange(len(signal_segment)) / self.fs
        axes[0].plot(t, signal_segment, 'b-', linewidth=0.8)
        axes[0].set_title('Original Signal')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        # Approximation
        t_approx = np.arange(len(approximation)) / self.fs * (2 ** level)
        axes[1].plot(t_approx, approximation, 'g-', linewidth=0.8)
        axes[1].set_title(f'Approximation (Level {level})')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True, alpha=0.3)
        
        # Details
        for i, detail in enumerate(details):
            t_detail = np.arange(len(detail)) / self.fs * (2 ** (level - i))
            axes[i+2].plot(t_detail, detail, 'r-', linewidth=0.8)
            axes[i+2].set_title(f'Detail Level {i+1}')
            axes[i+2].set_ylabel('Amplitude')
            axes[i+2].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (s)')
        
        plt.tight_layout()
        return fig


def demo_frequency_analysis():
    """
    Demonstrate frequency domain analysis
    """
    print("="*70)
    print("FREQUENCY DOMAIN ANALYSIS DEMONSTRATION")
    print("="*70)
    
    # Initialize analyzer
    fs = 360  # Hz
    analyzer = FrequencyAnalyzer(sampling_rate=fs)
    
    # Generate synthetic ECG
    duration = 10  # seconds
    t = np.arange(0, duration, 1/fs)
    
    # ECG with multiple frequency components
    ecg = np.zeros_like(t)
    heart_rate = 75  # BPM = 1.25 Hz
    beat_interval = 60 / heart_rate
    
    from scipy import signal
    for beat_time in np.arange(0.5, duration, beat_interval):
        beat_idx = int(beat_time * fs)
        if beat_idx < len(ecg) - 100:
            beat = gaussian(100, 10)
            ecg[beat_idx:beat_idx+100] += beat
    
    # Add some noise
    noise = 0.05 * np.random.randn(len(ecg))
    noisy_ecg = ecg + noise
    
    print(f"\n[Demo] Generated ECG signal: {duration}s @ {fs} Hz")
    
    # DFT vs FFT comparison
    perf_results = analyzer.compare_dft_fft(noisy_ecg, max_length=512)
    
    # Spectral analysis
    print("\n" + "="*70)
    print("SPECTRAL ANALYSIS")
    print("="*70)
    components = analyzer.identify_frequency_components(noisy_ecg[:2000])
    
    # Wavelet analysis
    print("\n" + "="*70)
    print("WAVELET ANALYSIS")
    print("="*70)
    approximation, details = analyzer.apply_dwt(noisy_ecg)
    features = analyzer.extract_wavelet_features(noisy_ecg)
    
    # Denoising
    denoised = analyzer.denoise_with_dwt(noisy_ecg)
    print(f"\n[Demo] Denoising SNR improvement: {10*np.log10(np.var(ecg)/np.var(noisy_ecg-ecg)):.2f} dB")
    
    print("\n[Demo] Frequency analysis demonstration completed!")
    print("="*70)
    
    return analyzer, noisy_ecg, denoised


if __name__ == "__main__":
    # Run demonstration
    analyzer, signal, denoised = demo_frequency_analysis()
    
    # Visualize
    plt.style.use('seaborn-v0_8-darkgrid')
    analyzer.plot_frequency_analysis(signal[:2000], duration=5.0)
    plt.savefig('/tmp/frequency_analysis_demo.png', dpi=150, bbox_inches='tight')
    print("\n[Demo] Saved frequency analysis to /tmp/frequency_analysis_demo.png")
    
    analyzer.plot_wavelet_decomposition(signal, level=5, duration=5.0)
    plt.savefig('/tmp/wavelet_decomposition_demo.png', dpi=150, bbox_inches='tight')
    print("[Demo] Saved wavelet decomposition to /tmp/wavelet_decomposition_demo.png")
    
    plt.show()
