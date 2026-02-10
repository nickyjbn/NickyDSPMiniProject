"""
Main Pipeline - ECG Signal Analysis System
Digital Signal Processing Mini-Project

Integrates all modules for end-to-end ECG analysis:
- Preprocessing (sampling, filtering, convolution)
- Frequency analysis (DFT/FFT/DWT)
- Image processing (scalogram analysis)
- Feature extraction and classification

Usage:
    python main.py [--record RECORD_ID] [--duration DURATION]
"""

import numpy as np
import matplotlib.pyplot as plt
import wfdb
import warnings
from scipy.signal.windows import gaussian
warnings.filterwarnings('ignore')

from ecg_preprocessing import ECGPreprocessor
from ecg_fft_dwt import FrequencyAnalyzer
from image_processing_ecg import ScalogramImageProcessor
from feature_extraction import ECGFeatureExtractor, ECGClassifier

import argparse
import os
from typing import Optional, Tuple


class ECGAnalysisPipeline:
    """
    Complete end-to-end ECG analysis pipeline
    Integrates all DSP concepts from theory and lab syllabus
    """
    
    def __init__(self, sampling_rate: float = 360.0):
        """
        Initialize pipeline with all components
        
        Args:
            sampling_rate: ECG sampling rate in Hz
        """
        self.fs = sampling_rate
        
        # Initialize all modules
        self.preprocessor = ECGPreprocessor(sampling_rate=sampling_rate)
        self.freq_analyzer = FrequencyAnalyzer(sampling_rate=sampling_rate)
        self.image_processor = ScalogramImageProcessor()
        self.feature_extractor = ECGFeatureExtractor(sampling_rate=sampling_rate)
        self.classifier = ECGClassifier()
        
        # Storage for results
        self.raw_signal = None
        self.filtered_signal = None
        self.r_peaks = None
        self.scalogram = None
        self.features = None
        self.classification = None
        
    def load_mit_bih_record(self, record_id: str = '100', 
                           duration: Optional[float] = None,
                           start_time: float = 0) -> Tuple[np.ndarray, dict]:
        """
        Load ECG signal from MIT-BIH Arrhythmia Database
        
        Args:
            record_id: MIT-BIH record ID (e.g., '100', '101')
            duration: Duration in seconds (None for full record)
            start_time: Start time in seconds
            
        Returns:
            Tuple of (ecg_signal, metadata)
        """
        print("="*70)
        print(f"LOADING MIT-BIH RECORD {record_id}")
        print("="*70)
        
        try:
            # Load record from PhysioNet
            record = wfdb.rdrecord(record_id, pn_dir='mitdb', 
                                  sampfrom=int(start_time * self.fs),
                                  sampto=int((start_time + duration) * self.fs) if duration else None)
            
            # Get ECG signal (usually lead 0)
            ecg_signal = record.p_signal[:, 0]
            
            metadata = {
                'record_id': record_id,
                'sampling_rate': record.fs,
                'duration': len(ecg_signal) / record.fs,
                'units': record.units[0] if record.units else 'mV',
                'comments': record.comments
            }
            
            print(f"[Dataset] Loaded record {record_id} from MIT-BIH database")
            print(f"[Dataset] Duration: {metadata['duration']:.1f} seconds")
            print(f"[Dataset] Sampling rate: {metadata['sampling_rate']} Hz")
            print(f"[Dataset] Samples: {len(ecg_signal)}")
            
            self.raw_signal = ecg_signal
            return ecg_signal, metadata
            
        except Exception as e:
            print(f"[Error] Could not load MIT-BIH record: {e}")
            print(f"[Info] Generating synthetic ECG signal instead...")
            return self._generate_synthetic_ecg(duration if duration else 10)
    
    def _generate_synthetic_ecg(self, duration: float = 10.0) -> Tuple[np.ndarray, dict]:
        """
        Generate synthetic ECG signal for demonstration
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Tuple of (ecg_signal, metadata)
        """
        from scipy import signal
        
        t = np.arange(0, duration, 1/self.fs)
        ecg = np.zeros_like(t)
        
        # Generate heartbeats
        heart_rate = 75  # BPM
        beat_interval = 60 / heart_rate
        
        for beat_time in np.arange(0.5, duration, beat_interval):
            beat_idx = int(beat_time * self.fs)
            if beat_idx < len(ecg) - 150:
                # Create PQRST complex
                # P wave
                p_wave = 0.25 * gaussian(30, 5)
                ecg[beat_idx-40:beat_idx-10] += p_wave
                
                # QRS complex
                qrs = gaussian(50, 8)
                qrs[20:30] *= -0.3  # Q wave
                ecg[beat_idx-5:beat_idx+45] += qrs
                
                # T wave
                t_wave = 0.35 * gaussian(60, 12)
                ecg[beat_idx+60:beat_idx+120] += t_wave
        
        # Add noise
        noise = 0.05 * np.random.randn(len(ecg))
        baseline_wander = 0.15 * np.sin(2 * np.pi * 0.3 * t)
        ecg_noisy = ecg + noise + baseline_wander
        
        metadata = {
            'record_id': 'synthetic',
            'sampling_rate': self.fs,
            'duration': duration,
            'units': 'normalized',
            'comments': ['Synthetic ECG signal for demonstration']
        }
        
        print(f"[Synthetic] Generated ECG: {duration} seconds @ {self.fs} Hz")
        
        self.raw_signal = ecg_noisy
        return ecg_noisy, metadata
    
    def run_complete_analysis(self, ecg_signal: np.ndarray, 
                             visualize: bool = True) -> dict:
        """
        Run complete end-to-end ECG analysis pipeline
        
        Args:
            ecg_signal: Raw ECG signal
            visualize: Whether to create visualizations
            
        Returns:
            Dictionary with all analysis results
        """
        print("\n" + "="*70)
        print("RUNNING COMPLETE ECG ANALYSIS PIPELINE")
        print("="*70)
        
        results = {}
        
        # =====================================================================
        # PHASE 1: PREPROCESSING (Unit 1, Lab Exp 1-2)
        # =====================================================================
        print("\n" + "-"*70)
        print("PHASE 1: SIGNAL PREPROCESSING")
        print("-"*70)
        
        # Normalize signal
        ecg_normalized = self.preprocessor.normalize_signal(ecg_signal)
        results['normalized_signal'] = ecg_normalized
        
        # Apply filtering
        filtered_signal = self.preprocessor.filter_ecg(
            ecg_normalized, method='filtfilt', lowcut=0.5, highcut=40.0
        )
        self.filtered_signal = filtered_signal
        results['filtered_signal'] = filtered_signal
        
        # Detect R-peaks
        r_peaks = self.preprocessor.detect_r_peaks_correlation(filtered_signal)
        if len(r_peaks) == 0:
            # Fallback to simple detection
            r_peaks = self.preprocessor.detect_r_peaks_simple(filtered_signal)
        self.r_peaks = r_peaks
        results['r_peaks'] = r_peaks
        
        # Calculate HRV
        if len(r_peaks) >= 2:
            hrv_metrics = self.preprocessor.calculate_hrv(r_peaks)
            results['hrv_metrics'] = hrv_metrics
        
        # =====================================================================
        # PHASE 2: FREQUENCY DOMAIN ANALYSIS (Unit 2, Lab Exp 3-4, 8)
        # =====================================================================
        print("\n" + "-"*70)
        print("PHASE 2: FREQUENCY DOMAIN ANALYSIS")
        print("-"*70)
        
        # DFT vs FFT comparison
        perf_results = self.freq_analyzer.compare_dft_fft(filtered_signal[:512])
        results['dft_fft_comparison'] = perf_results
        
        # Spectral analysis
        frequencies, magnitude = self.freq_analyzer.compute_magnitude_spectrum(filtered_signal)
        results['frequency_spectrum'] = (frequencies, magnitude)
        
        # Identify frequency components
        freq_components = self.freq_analyzer.identify_frequency_components(filtered_signal)
        results['frequency_components'] = freq_components
        
        # Wavelet transform
        approximation, details = self.freq_analyzer.apply_dwt(filtered_signal, level=5)
        results['dwt_coeffs'] = (approximation, details)
        
        # Denoise using wavelets
        denoised_signal = self.freq_analyzer.denoise_with_dwt(filtered_signal)
        results['denoised_signal'] = denoised_signal
        
        # =====================================================================
        # PHASE 3: SCALOGRAM GENERATION & IMAGE PROCESSING (Units 5-6, Lab Exp 7-10)
        # =====================================================================
        print("\n" + "-"*70)
        print("PHASE 3: SCALOGRAM IMAGE PROCESSING")
        print("-"*70)
        
        # Create scalogram
        scalogram, scalogram_freqs = self.image_processor.create_scalogram(
            filtered_signal, fs=self.fs
        )
        self.scalogram = scalogram
        results['scalogram'] = scalogram
        results['scalogram_frequencies'] = scalogram_freqs
        
        # Convert to image
        scalogram_image = self.image_processor.scalogram_to_uint8(scalogram)
        
        # Apply image processing techniques
        equalized = self.image_processor.histogram_equalization(scalogram_image)
        results['scalogram_equalized'] = equalized
        
        smoothed = self.image_processor.apply_gaussian_smoothing(scalogram_image)
        results['scalogram_smoothed'] = smoothed
        
        sharpened = self.image_processor.apply_laplacian_sharpening(scalogram_image)
        results['scalogram_sharpened'] = sharpened
        
        edges_sobel, _, _ = self.image_processor.sobel_edge_detection(scalogram_image)
        results['scalogram_edges'] = edges_sobel
        
        binary, threshold = self.image_processor.otsu_thresholding(scalogram_image)
        results['scalogram_binary'] = binary
        results['otsu_threshold'] = threshold
        
        # Extract texture features
        texture_features = self.image_processor.extract_texture_features_dwt(scalogram_image)
        results['texture_features'] = texture_features
        
        # =====================================================================
        # PHASE 4: FEATURE EXTRACTION & CLASSIFICATION
        # =====================================================================
        print("\n" + "-"*70)
        print("PHASE 4: FEATURE EXTRACTION & CLASSIFICATION")
        print("-"*70)
        
        # Extract all features
        features = self.feature_extractor.extract_all_features(
            filtered_signal, r_peaks, scalogram
        )
        self.features = features
        results['features'] = features
        
        # Classify
        classification, rule_triggers = self.classifier.classify_threshold_based(features)
        self.classification = classification
        results['classification'] = classification
        results['classification_rules'] = rule_triggers
        
        # =====================================================================
        # PHASE 5: VISUALIZATION
        # =====================================================================
        if visualize:
            print("\n" + "-"*70)
            print("PHASE 5: GENERATING VISUALIZATIONS")
            print("-"*70)
            self.create_comprehensive_visualization(results)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\n[Summary]")
        print(f"  Classification: {classification}")
        print(f"  Heart Rate: {results.get('hrv_metrics', {}).get('mean_hr_bpm', 'N/A'):.1f} BPM" if 'hrv_metrics' in results else "")
        print(f"  R-peaks detected: {len(r_peaks)}")
        print(f"  Total features: {len(features)}")
        print("="*70)
        
        return results
    
    def create_comprehensive_visualization(self, results: dict):
        """
        Create comprehensive visualization of all analysis stages
        
        Args:
            results: Dictionary with analysis results
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
        
        duration = 5.0  # Display first 5 seconds
        samples = int(duration * self.fs)
        t = np.arange(samples) / self.fs
        
        # Row 1: Time domain signals
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(t, self.raw_signal[:samples], 'b-', linewidth=0.7)
        ax1.set_title('1. Original ECG Signal', fontsize=10, fontweight='bold')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(t, self.filtered_signal[:samples], 'g-', linewidth=0.7)
        if self.r_peaks is not None:
            r_in_range = self.r_peaks[self.r_peaks < samples]
            ax2.plot(r_in_range / self.fs, self.filtered_signal[r_in_range], 
                    'ro', markersize=6)
        ax2.set_title('2. Filtered + R-peaks (Unit 1)', fontsize=10, fontweight='bold')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[0, 2])
        if 'hrv_metrics' in results:
            hrv = results['hrv_metrics']
            metrics_text = f"Heart Rate: {hrv['mean_hr_bpm']:.1f} BPM\n"
            metrics_text += f"Mean RR: {hrv['mean_rr_ms']:.1f} ms\n"
            metrics_text += f"SDNN: {hrv['sdnn_ms']:.1f} ms\n"
            metrics_text += f"RMSSD: {hrv['rmssd_ms']:.1f} ms\n"
            metrics_text += f"\nClassification:\n{self.classification}"
        else:
            metrics_text = f"Classification:\n{self.classification}"
        
        ax3.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax3.axis('off')
        ax3.set_title('3. HRV Metrics', fontsize=10, fontweight='bold')
        
        # Row 2: Frequency domain
        ax4 = fig.add_subplot(gs[1, 0])
        freqs, mags = results['frequency_spectrum']
        ax4.plot(freqs, mags, 'r-', linewidth=0.7)
        ax4.set_xlim([0, 50])
        ax4.set_title('4. Magnitude Spectrum (Unit 2)', fontsize=10, fontweight='bold')
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Magnitude')
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(gs[1, 1])
        from scipy import signal as sp_signal
        f, t_spec, Sxx = sp_signal.spectrogram(self.filtered_signal[:samples], 
                                                self.fs, nperseg=256, noverlap=128)
        im = ax5.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), 
                           shading='gouraud', cmap='viridis')
        ax5.set_ylim([0, 50])
        ax5.set_title('5. Spectrogram (FFT-based)', fontsize=10, fontweight='bold')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Frequency (Hz)')
        plt.colorbar(im, ax=ax5, label='Power (dB)')
        
        ax6 = fig.add_subplot(gs[1, 2])
        if 'dft_fft_comparison' in results:
            comp = results['dft_fft_comparison']
            labels = ['DFT', 'FFT\n(Custom)', 'FFT\n(NumPy)']
            times = [comp['dft_time'], comp['fft_custom_time'], comp['fft_numpy_time']]
            bars = ax6.bar(labels, times, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
            ax6.set_ylabel('Time (seconds)')
            ax6.set_title('6. DFT vs FFT Performance (Lab Exp 3-4)', 
                         fontsize=10, fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='y')
            
            # Add speedup annotations
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{times[i]:.4f}s', ha='center', va='bottom', fontsize=8)
        
        # Row 3: Scalogram and image processing
        ax7 = fig.add_subplot(gs[2, 0])
        im = ax7.imshow(self.scalogram, aspect='auto', cmap='viridis', 
                       extent=[0, duration, 0, 50])
        ax7.set_title('7. Scalogram (Unit 2.5)', fontsize=10, fontweight='bold')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Frequency (Hz)')
        plt.colorbar(im, ax=ax7, label='Magnitude')
        
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.imshow(results['scalogram_equalized'], cmap='gray', aspect='auto')
        ax8.set_title('8. Histogram Equalization (Unit 5.1)', 
                     fontsize=10, fontweight='bold')
        ax8.axis('off')
        
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.imshow(results['scalogram_edges'], cmap='gray', aspect='auto')
        ax9.set_title('9. Edge Detection - Sobel (Lab Exp 10)', 
                     fontsize=10, fontweight='bold')
        ax9.axis('off')
        
        # Row 4: More image processing and features
        ax10 = fig.add_subplot(gs[3, 0])
        ax10.imshow(results['scalogram_sharpened'], cmap='gray', aspect='auto')
        ax10.set_title('10. Laplacian Sharpening (Lab Exp 9)', 
                      fontsize=10, fontweight='bold')
        ax10.axis('off')
        
        ax11 = fig.add_subplot(gs[3, 1])
        ax11.imshow(results['scalogram_binary'], cmap='gray', aspect='auto')
        ax11.set_title(f"11. Otsu Threshold (Unit 6, T={results['otsu_threshold']:.0f})", 
                      fontsize=10, fontweight='bold')
        ax11.axis('off')
        
        ax12 = fig.add_subplot(gs[3, 2])
        # Display top features
        feature_items = list(self.features.items())[:8]
        feature_text = "Top Features:\n\n"
        for key, value in feature_items:
            feature_text += f"{key}: {value:.3f}\n"
        ax12.text(0.05, 0.95, feature_text, fontsize=8, verticalalignment='top',
                 family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax12.axis('off')
        ax12.set_title('12. Extracted Features', fontsize=10, fontweight='bold')
        
        # Overall title
        fig.suptitle('ECG Signal Analysis Pipeline - Complete Visualization', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        # Save figure
        output_path = '/tmp/ecg_complete_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[Visualization] Saved to {output_path}")
        
        return fig


def main():
    """
    Main entry point for ECG analysis pipeline
    """
    parser = argparse.ArgumentParser(
        description='ECG Signal Analysis System - DSP Mini-Project'
    )
    parser.add_argument('--record', type=str, default='100',
                       help='MIT-BIH record ID (default: 100)')
    parser.add_argument('--duration', type=float, default=10.0,
                       help='Duration in seconds (default: 10.0)')
    parser.add_argument('--start', type=float, default=0.0,
                       help='Start time in seconds (default: 0.0)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    print("\n" + "="*70)
    print("ECG SIGNAL ANALYSIS SYSTEM")
    print("Digital Signal Processing Mini-Project")
    print("="*70)
    
    pipeline = ECGAnalysisPipeline(sampling_rate=360.0)
    
    # Load data
    ecg_signal, metadata = pipeline.load_mit_bih_record(
        record_id=args.record,
        duration=args.duration,
        start_time=args.start
    )
    
    # Run analysis
    results = pipeline.run_complete_analysis(
        ecg_signal,
        visualize=not args.no_viz
    )
    
    # Show plots
    if not args.no_viz:
        plt.show()
    
    print("\n[Pipeline] Analysis complete! Check /tmp/ for visualizations.")
    
    return pipeline, results


if __name__ == "__main__":
    pipeline, results = main()
