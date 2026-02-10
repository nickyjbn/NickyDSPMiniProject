"""
Image Processing on ECG Scalograms Module
Digital Signal Processing Mini-Project

Implements:
- Unit 5: Spatial/Frequency Domain Filtering
- Unit 6: Image Segmentation
- Lab Exp 7: Histogram Processing
- Lab Exp 8: DCT/DWT on Images
- Lab Exp 9: Smoothing/Sharpening
- Lab Exp 10: Edge Detection (Sobel/Prewitt)
"""

import numpy as np
import cv2
import pywt
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import filters, exposure, transform
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from typing import Tuple, Optional


class ScalogramImageProcessor:
    """
    Apply image processing techniques to ECG scalograms
    Treat time-frequency representation as a 2D grayscale image
    """
    
    def __init__(self):
        """Initialize image processor"""
        pass
    
    # =========================================================================
    # Scalogram Generation
    # =========================================================================
    
    def create_scalogram(self, signal: np.ndarray, fs: float = 360.0,
                        wavelet: str = 'morl', scales: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time-frequency scalogram from 1D ECG signal
        
        Args:
            signal: Input ECG signal
            fs: Sampling frequency
            wavelet: Wavelet type
            scales: Array of scales
            
        Returns:
            Tuple of (scalogram_image, frequencies)
        """
        if scales is None:
            scales = np.arange(1, 128)
        
        # Compute CWT
        coefficients, frequencies = pywt.cwt(signal, scales, wavelet, 1/fs)
        
        # Convert to magnitude (treat as grayscale image)
        scalogram = np.abs(coefficients)
        
        print(f"[Scalogram] Generated {scalogram.shape} time-frequency representation")
        
        return scalogram, frequencies
    
    def scalogram_to_uint8(self, scalogram: np.ndarray) -> np.ndarray:
        """
        Convert scalogram to 8-bit grayscale image
        
        Args:
            scalogram: Float scalogram
            
        Returns:
            8-bit grayscale image
        """
        # Normalize to [0, 255]
        normalized = (scalogram - scalogram.min()) / (scalogram.max() - scalogram.min())
        image = (normalized * 255).astype(np.uint8)
        
        return image
    
    # =========================================================================
    # Unit 5.1 & Lab Exp 7: Histogram Processing
    # =========================================================================
    
    def compute_histogram(self, image: np.ndarray, bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute histogram of grayscale image
        
        Lab Exp 7: Histogram Processing
        
        Args:
            image: Grayscale image
            bins: Number of histogram bins
            
        Returns:
            Tuple of (histogram, bin_centers)
        """
        hist, bin_edges = np.histogram(image.ravel(), bins=bins, range=(0, 256))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        print(f"[Lab Exp 7] Computed histogram with {bins} bins")
        
        return hist, bin_centers
    
    def contrast_stretching(self, image: np.ndarray, 
                           percentile_low: float = 2, 
                           percentile_high: float = 98) -> np.ndarray:
        """
        Apply contrast stretching to enhance image contrast
        
        Lab Exp 7: Contrast Stretching
        
        Args:
            image: Input grayscale image
            percentile_low: Lower percentile for stretching
            percentile_high: Higher percentile for stretching
            
        Returns:
            Contrast-stretched image
        """
        # Find intensity values at percentiles
        p_low, p_high = np.percentile(image, (percentile_low, percentile_high))
        
        # Stretch intensities
        stretched = exposure.rescale_intensity(image, in_range=(p_low, p_high))
        
        print(f"[Lab Exp 7] Applied contrast stretching: [{p_low:.1f}, {p_high:.1f}] -> [0, 255]")
        
        return stretched
    
    def histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization for contrast enhancement
        
        Unit 5.1: Histogram Equalization
        Lab Exp 7: Histogram Processing
        
        Redistributes intensity values to achieve uniform histogram
        
        Args:
            image: Input grayscale image (uint8)
            
        Returns:
            Equalized image
        """
        # Histogram equalization
        equalized = exposure.equalize_hist(image)
        
        # Convert back to uint8
        equalized_uint8 = (equalized * 255).astype(np.uint8)
        
        print(f"[Unit 5.1] Applied histogram equalization")
        print(f"[Lab Exp 7] Contrast enhanced using histogram redistribution")
        
        return equalized_uint8
    
    def adaptive_histogram_equalization(self, image: np.ndarray, 
                                       clip_limit: float = 0.03) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Unit 5.1: Advanced histogram equalization
        
        Args:
            image: Input grayscale image
            clip_limit: Clipping limit for contrast
            
        Returns:
            CLAHE-enhanced image
        """
        # CLAHE
        equalized = exposure.equalize_adapthist(image / 255.0, clip_limit=clip_limit)
        equalized_uint8 = (equalized * 255).astype(np.uint8)
        
        print(f"[Unit 5.1] Applied CLAHE (Adaptive Histogram Equalization)")
        
        return equalized_uint8
    
    # =========================================================================
    # Unit 5.2 & Lab Exp 9: Smoothing/Sharpening Filters
    # =========================================================================
    
    def apply_gaussian_smoothing(self, image: np.ndarray, 
                                 kernel_size: int = 5, 
                                 sigma: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian smoothing filter
        
        Lab Exp 9: Smoothing Filters
        Unit 5.2: Low-pass filtering for noise reduction
        
        Args:
            image: Input image
            kernel_size: Size of Gaussian kernel
            sigma: Standard deviation of Gaussian
            
        Returns:
            Smoothed image
        """
        # Gaussian blur using OpenCV
        smoothed = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        print(f"[Lab Exp 9] Applied Gaussian smoothing (kernel={kernel_size}, sigma={sigma})")
        print(f"[Unit 5.2] Low-pass filtering reduces noise")
        
        return smoothed
    
    def apply_median_filter(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply median filter for noise removal
        
        Lab Exp 9: Non-linear smoothing
        Effective for salt-and-pepper noise
        
        Args:
            image: Input image
            kernel_size: Size of median filter kernel
            
        Returns:
            Filtered image
        """
        filtered = cv2.medianBlur(image, kernel_size)
        
        print(f"[Lab Exp 9] Applied median filter (kernel={kernel_size})")
        
        return filtered
    
    def apply_laplacian_sharpening(self, image: np.ndarray, 
                                   alpha: float = 0.5) -> np.ndarray:
        """
        Apply Laplacian sharpening to enhance edges
        
        Lab Exp 9: Sharpening Filters
        Unit 5.2: High-pass filtering emphasizes edges
        
        Sharpened = Original + alpha * Laplacian(Original)
        
        Args:
            image: Input image
            alpha: Sharpening strength
            
        Returns:
            Sharpened image
        """
        # Compute Laplacian (second derivative)
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        
        # Sharpen: add scaled Laplacian to original
        sharpened = image.astype(float) + alpha * laplacian
        
        # Clip to valid range
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        print(f"[Lab Exp 9] Applied Laplacian sharpening (alpha={alpha})")
        print(f"[Unit 5.2] High-pass filtering enhances edges")
        
        return sharpened
    
    def unsharp_masking(self, image: np.ndarray, 
                       sigma: float = 1.0, 
                       amount: float = 1.5) -> np.ndarray:
        """
        Apply unsharp masking for sharpening
        
        Lab Exp 9: Unsharp Masking
        Sharpened = Original + amount * (Original - Blurred)
        
        Args:
            image: Input image
            sigma: Gaussian blur sigma
            amount: Sharpening amount
            
        Returns:
            Sharpened image
        """
        # Blur image
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        
        # Unsharp mask = Original - Blurred
        unsharp_mask = image.astype(float) - blurred.astype(float)
        
        # Sharpened = Original + amount * mask
        sharpened = image.astype(float) + amount * unsharp_mask
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        print(f"[Lab Exp 9] Applied unsharp masking (sigma={sigma}, amount={amount})")
        
        return sharpened
    
    # =========================================================================
    # Unit 5.2: Gradient-based Edge Enhancement
    # =========================================================================
    
    def compute_gradient_magnitude(self, image: np.ndarray) -> np.ndarray:
        """
        Compute gradient magnitude using Sobel operators
        
        Unit 5.2: Gradient computation
        |∇I| = sqrt(Gx^2 + Gy^2)
        
        Args:
            image: Input image
            
        Returns:
            Gradient magnitude
        """
        # Compute gradients in x and y directions
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
        
        print(f"[Unit 5.2] Computed gradient magnitude")
        
        return magnitude
    
    # =========================================================================
    # Lab Exp 10 & Unit 6: Edge Detection
    # =========================================================================
    
    def sobel_edge_detection(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply Sobel edge detection
        
        Lab Exp 10: Edge Detection (Sobel)
        Unit 6: Edge detection for segmentation
        
        Sobel masks:
        Gx = [[-1, 0, 1],      Gy = [[-1, -2, -1],
              [-2, 0, 2],             [ 0,  0,  0],
              [-1, 0, 1]]             [ 1,  2,  1]]
        
        Args:
            image: Input grayscale image
            
        Returns:
            Tuple of (edges, gradient_x, gradient_y)
        """
        # Sobel operators
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Edge magnitude
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        edges = np.clip(edges, 0, 255).astype(np.uint8)
        
        print(f"[Lab Exp 10] Applied Sobel edge detection")
        print(f"[Unit 6] Edges highlight transient features (P, QRS, T waves)")
        
        return edges, sobel_x, sobel_y
    
    def prewitt_edge_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Prewitt edge detection
        
        Lab Exp 10: Edge Detection (Prewitt)
        
        Prewitt masks:
        Gx = [[-1, 0, 1],      Gy = [[-1, -1, -1],
              [-1, 0, 1],             [ 0,  0,  0],
              [-1, 0, 1]]             [ 1,  1,  1]]
        
        Args:
            image: Input grayscale image
            
        Returns:
            Edge image
        """
        # Prewitt kernels
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        
        # Convolve with Prewitt kernels
        prewitt_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
        prewitt_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)
        
        # Edge magnitude
        edges = np.sqrt(prewitt_x**2 + prewitt_y**2)
        edges = np.clip(edges, 0, 255).astype(np.uint8)
        
        print(f"[Lab Exp 10] Applied Prewitt edge detection")
        
        return edges
    
    def canny_edge_detection(self, image: np.ndarray, 
                            low_threshold: float = 50, 
                            high_threshold: float = 150) -> np.ndarray:
        """
        Apply Canny edge detection
        
        Lab Exp 10: Advanced edge detection
        Multi-stage algorithm: Gaussian smoothing, gradient, non-max suppression, hysteresis
        
        Args:
            image: Input image
            low_threshold: Low threshold for hysteresis
            high_threshold: High threshold for hysteresis
            
        Returns:
            Binary edge map
        """
        edges = cv2.Canny(image, low_threshold, high_threshold)
        
        print(f"[Lab Exp 10] Applied Canny edge detection")
        
        return edges
    
    # =========================================================================
    # Unit 6: Image Segmentation - Thresholding
    # =========================================================================
    
    def otsu_thresholding(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply Otsu's automatic thresholding
        
        Unit 6: Otsu's method for optimal threshold selection
        Minimizes intra-class variance (or maximizes inter-class variance)
        
        Args:
            image: Input grayscale image
            
        Returns:
            Tuple of (binary_image, threshold_value)
        """
        # Otsu's thresholding
        threshold_value, binary_image = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        print(f"[Unit 6] Applied Otsu's thresholding")
        print(f"[Unit 6] Optimal threshold: {threshold_value:.1f}")
        print(f"[Unit 6] Isolates high-energy regions in scalogram")
        
        return binary_image, threshold_value
    
    def adaptive_thresholding(self, image: np.ndarray, 
                             block_size: int = 11, 
                             C: int = 2) -> np.ndarray:
        """
        Apply adaptive thresholding
        
        Unit 6: Local thresholding for varying illumination
        
        Args:
            image: Input image
            block_size: Size of neighborhood
            C: Constant subtracted from mean
            
        Returns:
            Binary image
        """
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, C
        )
        
        print(f"[Unit 6] Applied adaptive thresholding (block_size={block_size})")
        
        return binary
    
    # =========================================================================
    # Unit 6: Hough Transform for Line Detection
    # =========================================================================
    
    def hough_line_detection(self, edge_image: np.ndarray, 
                            threshold: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply Hough transform to detect lines (periodic R-peak patterns)
        
        Unit 6: Hough Transform
        Detects quasi-periodic patterns in scalogram (R-peaks show as vertical lines)
        
        Args:
            edge_image: Binary edge image
            threshold: Minimum number of votes for line detection
            
        Returns:
            Tuple of (lines_image, angles, distances)
        """
        # Ensure binary edge image
        if edge_image.dtype != np.uint8:
            edge_image = edge_image.astype(np.uint8)
        
        # Hough line transform
        lines = cv2.HoughLines(edge_image, 1, np.pi/180, threshold)
        
        # Create image with detected lines
        lines_image = np.zeros_like(edge_image)
        
        angles = []
        distances = []
        
        if lines is not None:
            for line in lines[:10]:  # Limit to top 10 lines
                rho, theta = line[0]
                angles.append(theta)
                distances.append(rho)
                
                # Convert to Cartesian coordinates
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                
                cv2.line(lines_image, (x1, y1), (x2, y2), 255, 2)
        
        print(f"[Unit 6] Applied Hough transform")
        print(f"[Unit 6] Detected {len(angles)} lines (periodic R-peak patterns)")
        
        return lines_image, np.array(angles), np.array(distances)
    
    # =========================================================================
    # Lab Exp 8: DCT/DWT on Images
    # =========================================================================
    
    def apply_dct_2d(self, image: np.ndarray) -> np.ndarray:
        """
        Apply 2D Discrete Cosine Transform
        
        Lab Exp 8: DCT on Images
        Used for compression and feature extraction
        
        Args:
            image: Input image
            
        Returns:
            DCT coefficients
        """
        # Convert to float
        image_float = image.astype(float)
        
        # 2D DCT
        dct_coeffs = cv2.dct(image_float)
        
        print(f"[Lab Exp 8] Applied 2D DCT")
        
        return dct_coeffs
    
    def apply_dwt_2d(self, image: np.ndarray, wavelet: str = 'haar') -> Tuple:
        """
        Apply 2D Discrete Wavelet Transform
        
        Lab Exp 8: DWT on Images
        Decomposes image into approximation and detail coefficients
        
        Args:
            image: Input image
            wavelet: Wavelet type
            
        Returns:
            Tuple of (LL, (LH, HL, HH)) coefficients
        """
        # 2D DWT
        coeffs = pywt.dwt2(image, wavelet)
        LL, (LH, HL, HH) = coeffs
        
        print(f"[Lab Exp 8] Applied 2D DWT with {wavelet} wavelet")
        print(f"[Lab Exp 8] Approximation (LL): {LL.shape}")
        print(f"[Lab Exp 8] Details: LH={LH.shape}, HL={HL.shape}, HH={HH.shape}")
        
        return coeffs
    
    def extract_texture_features_dct(self, image: np.ndarray, 
                                     block_size: int = 8) -> dict:
        """
        Extract texture features from DCT coefficients
        
        Lab Exp 8: Feature extraction from DCT
        
        Args:
            image: Input image
            block_size: Size of DCT blocks
            
        Returns:
            Dictionary of texture features
        """
        # Divide image into blocks and compute DCT
        h, w = image.shape
        num_blocks_h = h // block_size
        num_blocks_w = w // block_size
        
        energies = []
        entropies = []
        
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                block = image[i*block_size:(i+1)*block_size, 
                             j*block_size:(j+1)*block_size]
                
                # DCT of block
                dct_block = cv2.dct(block.astype(float))
                
                # Energy
                energy = np.sum(dct_block ** 2)
                energies.append(energy)
                
                # Entropy (approximate)
                hist, _ = np.histogram(dct_block.ravel(), bins=50)
                hist = hist / hist.sum()
                hist = hist[hist > 0]
                entropy = -np.sum(hist * np.log2(hist))
                entropies.append(entropy)
        
        features = {
            'mean_energy': np.mean(energies),
            'std_energy': np.std(energies),
            'mean_entropy': np.mean(entropies),
            'std_entropy': np.std(entropies)
        }
        
        print(f"[Lab Exp 8] Extracted texture features from DCT")
        
        return features
    
    def extract_texture_features_dwt(self, image: np.ndarray, 
                                    wavelet: str = 'haar') -> dict:
        """
        Extract texture features from DWT coefficients
        
        Lab Exp 8: Feature extraction from DWT
        
        Args:
            image: Input image
            wavelet: Wavelet type
            
        Returns:
            Dictionary of texture features
        """
        # 2D DWT
        LL, (LH, HL, HH) = self.apply_dwt_2d(image, wavelet)
        
        features = {
            'LL_energy': np.sum(LL ** 2),
            'LH_energy': np.sum(LH ** 2),
            'HL_energy': np.sum(HL ** 2),
            'HH_energy': np.sum(HH ** 2),
            'LL_mean': np.mean(LL),
            'LH_mean': np.mean(LH),
            'HL_mean': np.mean(HL),
            'HH_mean': np.mean(HH),
            'LL_std': np.std(LL),
            'LH_std': np.std(LH),
            'HL_std': np.std(HL),
            'HH_std': np.std(HH),
        }
        
        print(f"[Lab Exp 8] Extracted texture features from DWT")
        
        return features
    
    # =========================================================================
    # Visualization
    # =========================================================================
    
    def plot_image_processing_pipeline(self, scalogram: np.ndarray):
        """
        Visualize complete image processing pipeline
        
        Args:
            scalogram: Input scalogram
        """
        # Convert to uint8
        image = self.scalogram_to_uint8(scalogram)
        
        # Apply various processing steps
        equalized = self.histogram_equalization(image)
        smoothed = self.apply_gaussian_smoothing(image, kernel_size=5)
        sharpened = self.apply_laplacian_sharpening(image, alpha=0.5)
        edges_sobel, _, _ = self.sobel_edge_detection(image)
        edges_prewitt = self.prewitt_edge_detection(image)
        binary, threshold = self.otsu_thresholding(image)
        
        # Create figure
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        # Original
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Scalogram')
        axes[0, 0].axis('off')
        
        # Histogram
        hist, bins = self.compute_histogram(image)
        axes[0, 1].plot(bins, hist, 'b-')
        axes[0, 1].set_title('Histogram')
        axes[0, 1].set_xlabel('Intensity')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram Equalization
        axes[0, 2].imshow(equalized, cmap='gray')
        axes[0, 2].set_title('Histogram Equalization\n(Unit 5.1, Lab Exp 7)')
        axes[0, 2].axis('off')
        
        # Gaussian Smoothing
        axes[1, 0].imshow(smoothed, cmap='gray')
        axes[1, 0].set_title('Gaussian Smoothing\n(Lab Exp 9)')
        axes[1, 0].axis('off')
        
        # Laplacian Sharpening
        axes[1, 1].imshow(sharpened, cmap='gray')
        axes[1, 1].set_title('Laplacian Sharpening\n(Lab Exp 9)')
        axes[1, 1].axis('off')
        
        # Sobel Edge Detection
        axes[1, 2].imshow(edges_sobel, cmap='gray')
        axes[1, 2].set_title('Sobel Edge Detection\n(Lab Exp 10)')
        axes[1, 2].axis('off')
        
        # Prewitt Edge Detection
        axes[2, 0].imshow(edges_prewitt, cmap='gray')
        axes[2, 0].set_title('Prewitt Edge Detection\n(Lab Exp 10)')
        axes[2, 0].axis('off')
        
        # Otsu's Thresholding
        axes[2, 1].imshow(binary, cmap='gray')
        axes[2, 1].set_title(f"Otsu's Thresholding\n(Unit 6, T={threshold:.1f})")
        axes[2, 1].axis('off')
        
        # DWT visualization
        LL, (LH, HL, HH) = self.apply_dwt_2d(image)
        dwt_combined = np.vstack([
            np.hstack([LL, LH]),
            np.hstack([HL, HH])
        ])
        axes[2, 2].imshow(dwt_combined, cmap='gray')
        axes[2, 2].set_title('2D DWT Decomposition\n(Lab Exp 8)')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        return fig


def demo_image_processing():
    """
    Demonstrate image processing on ECG scalogram
    """
    print("="*70)
    print("IMAGE PROCESSING ON ECG SCALOGRAMS DEMONSTRATION")
    print("="*70)
    
    # Create synthetic scalogram-like image
    processor = ScalogramImageProcessor()
    
    # Generate synthetic ECG
    fs = 360
    duration = 10
    t = np.arange(0, duration, 1/fs)
    
    from scipy import signal
    ecg = np.zeros_like(t)
    for beat_time in np.arange(0.5, duration, 0.8):
        beat_idx = int(beat_time * fs)
        if beat_idx < len(ecg) - 100:
            beat = signal.gaussian(100, 10)
            ecg[beat_idx:beat_idx+100] += beat
    
    # Create scalogram
    scalogram, frequencies = processor.create_scalogram(ecg, fs)
    
    print(f"\n[Demo] Created scalogram: {scalogram.shape}")
    
    # Convert to uint8 image
    image = processor.scalogram_to_uint8(scalogram)
    
    # Test various processing techniques
    print("\n" + "="*70)
    print("HISTOGRAM PROCESSING (Unit 5.1, Lab Exp 7)")
    print("="*70)
    equalized = processor.histogram_equalization(image)
    
    print("\n" + "="*70)
    print("SMOOTHING/SHARPENING (Unit 5.2, Lab Exp 9)")
    print("="*70)
    smoothed = processor.apply_gaussian_smoothing(image)
    sharpened = processor.apply_laplacian_sharpening(image)
    
    print("\n" + "="*70)
    print("EDGE DETECTION (Unit 6, Lab Exp 10)")
    print("="*70)
    edges_sobel, _, _ = processor.sobel_edge_detection(image)
    edges_prewitt = processor.prewitt_edge_detection(image)
    
    print("\n" + "="*70)
    print("SEGMENTATION (Unit 6)")
    print("="*70)
    binary, threshold = processor.otsu_thresholding(image)
    
    print("\n" + "="*70)
    print("DCT/DWT ON IMAGES (Lab Exp 8)")
    print("="*70)
    dct_features = processor.extract_texture_features_dct(image)
    dwt_features = processor.extract_texture_features_dwt(image)
    
    print("\n[Demo] Image processing demonstration completed!")
    print("="*70)
    
    return processor, scalogram, image


if __name__ == "__main__":
    # Run demonstration
    processor, scalogram, image = demo_image_processing()
    
    # Visualize
    plt.style.use('seaborn-v0_8-darkgrid')
    processor.plot_image_processing_pipeline(scalogram)
    plt.savefig('/tmp/image_processing_demo.png', dpi=150, bbox_inches='tight')
    print("\n[Demo] Saved visualization to /tmp/image_processing_demo.png")
    plt.show()
