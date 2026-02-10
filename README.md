# ECG Signal Analysis System

**Digital Signal Processing and Applications Mini-Project**

A comprehensive Python-based ECG signal analysis system integrating concepts from DSP theory (DJS22CEC802) and lab syllabus (DJS22CEL802) for Computer Engineering B.Tech final-year students.

## 📋 Project Overview

This project implements a complete ECG signal analysis pipeline covering:
- **Time-domain analysis**: Sampling, filtering, convolution, R-peak detection
- **Frequency-domain analysis**: DFT/FFT/Wavelet transforms
- **Image processing**: Scalogram analysis with histogram equalization, edge detection
- **Feature extraction & classification**: HRV metrics, spectral features, texture analysis

## 🎯 Syllabus Coverage

### Theory Syllabus (DJS22CEC802)
- ✅ **Unit 1**: Discrete-Time Signals & Systems, Linear Convolution, Impulse Response
- ✅ **Unit 2**: DFT/FFT, Radix-2 DIT-FFT, Wavelet Transform
- ✅ **Unit 3**: Overlap-Add/Save Methods, FFT-based Filtering
- ✅ **Unit 5**: Histogram Equalization, Smoothing/Sharpening, Gradient/Laplacian
- ✅ **Unit 6**: Edge Detection, Otsu's Thresholding, Hough Transform

### Lab Experiments (DJS22CEL802)
- ✅ **Exp 1**: Sampling & Reconstruction
- ✅ **Exp 2**: Discrete Correlation & Convolution
- ✅ **Exp 3**: DFT Implementation
- ✅ **Exp 4**: FFT Implementation
- ✅ **Exp 7**: Histogram Processing
- ✅ **Exp 8**: DFT/DCT/DWT on Images
- ✅ **Exp 9**: Smoothing/Sharpening Filters
- ✅ **Exp 10**: Edge Detection (Sobel/Prewitt)

## 📁 Project Structure

```
NickyDSPMiniProject/
├── ecg_preprocessing.py          # Signal preprocessing (Unit 1, Lab Exp 1-2)
├── ecg_fft_dwt.py               # Frequency analysis (Unit 2, Lab Exp 3-4, 8)
├── image_processing_ecg.py       # Scalogram processing (Units 5-6, Lab Exp 7-10)
├── feature_extraction.py         # Feature extraction & classification
├── main.py                       # Complete pipeline integration
├── ECG_Analysis_Tutorial.ipynb  # Interactive tutorial
├── REPORT.md                     # Detailed syllabus mapping
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/nickyjbn/NickyDSPMiniProject.git
cd NickyDSPMiniProject
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Basic Usage

**Run complete pipeline with MIT-BIH dataset:**
```bash
python main.py --record 100 --duration 10
```

**Run with synthetic ECG (if MIT-BIH download fails):**
```bash
python main.py --duration 10
```

**Command-line options:**
```bash
python main.py --help
```
- `--record`: MIT-BIH record ID (default: 100)
- `--duration`: Duration in seconds (default: 10.0)
- `--start`: Start time in seconds (default: 0.0)
- `--no-viz`: Disable visualization

### Module Examples

**1. ECG Preprocessing:**
```python
from ecg_preprocessing import ECGPreprocessor

preprocessor = ECGPreprocessor(sampling_rate=360)
filtered = preprocessor.filter_ecg(ecg_signal)
r_peaks = preprocessor.detect_r_peaks_correlation(filtered)
hrv_metrics = preprocessor.calculate_hrv(r_peaks)
```

**2. Frequency Analysis:**
```python
from ecg_fft_dwt import FrequencyAnalyzer

analyzer = FrequencyAnalyzer(sampling_rate=360)
# DFT vs FFT comparison
results = analyzer.compare_dft_fft(signal)
# Wavelet transform
approx, details = analyzer.apply_dwt(signal, level=5)
```

**3. Image Processing on Scalogram:**
```python
from image_processing_ecg import ScalogramImageProcessor

processor = ScalogramImageProcessor()
scalogram, freqs = processor.create_scalogram(signal, fs=360)
equalized = processor.histogram_equalization(scalogram_image)
edges = processor.sobel_edge_detection(scalogram_image)
```

**4. Feature Extraction:**
```python
from feature_extraction import ECGFeatureExtractor, ECGClassifier

extractor = ECGFeatureExtractor(sampling_rate=360)
features = extractor.extract_all_features(signal, r_peaks, scalogram)
classifier = ECGClassifier()
classification, rules = classifier.classify_threshold_based(features)
```

## 📊 Features Implemented

### Time-Domain Features
- QRS width, amplitude, and area
- ST-segment elevation/depression
- T-wave amplitude
- Heart rate variability (SDNN, RMSSD, pNN50)
- Statistical features (mean, std, skewness, kurtosis)

### Frequency-Domain Features
- Magnitude and power spectrum
- VLF, LF, HF power bands
- LF/HF ratio
- Spectral entropy
- Wavelet coefficients (approximation + details)
- Wavelet energy and entropy

### Image-Based Features (from Scalogram)
- Texture energy and entropy
- Gradient-based features
- DCT/DWT texture features

## 🔬 Dataset

The project uses the **MIT-BIH Arrhythmia Database** from PhysioNet:
- 48 ECG recordings (30 minutes each)
- 360 Hz sampling rate
- Annotated R-peaks and arrhythmias
- Accessible via `wfdb` Python library

**Common records:**
- `100`: Normal sinus rhythm
- `101`: Atrial premature beats
- `106`: Ventricular tachycardia
- `200`: Various arrhythmias

If dataset download fails, the system automatically generates synthetic ECG signals.

## 📈 Outputs

The pipeline generates:
1. **Comprehensive visualization** (`/tmp/ecg_complete_analysis.png`):
   - Original and filtered signals
   - R-peaks and HRV metrics
   - Magnitude spectrum and spectrogram
   - Scalogram with various image processing stages
   - DFT vs FFT performance comparison

2. **Console output** with:
   - Processing steps with syllabus references
   - Performance metrics
   - Classification results

3. **Jupyter notebook** with interactive exploration

## 🧪 Testing Individual Modules

Each module can be run standalone for testing:

```bash
python ecg_preprocessing.py        # Demo preprocessing
python ecg_fft_dwt.py             # Demo frequency analysis
python image_processing_ecg.py    # Demo image processing
python feature_extraction.py      # Demo feature extraction
```

## 📚 Documentation

- **README.md** (this file): Setup and usage guide
- **REPORT.md**: Detailed syllabus mapping and implementation details
- **ECG_Analysis_Tutorial.ipynb**: Interactive tutorial with theory

## 🛠️ Technical Stack

- **NumPy**: Array operations and numerical computing
- **SciPy**: Signal processing, filters, spectral analysis
- **Matplotlib**: Visualization and plotting
- **PyWavelets**: Wavelet transforms (DWT, CWT)
- **WFDB**: MIT-BIH database access
- **OpenCV**: Image processing operations
- **Scikit-learn**: Machine learning (SVM classifier)
- **Scikit-image**: Advanced image processing

## 🎓 Key Algorithms Implemented

1. **Radix-2 DIT-FFT** (from scratch) - O(N log N) complexity
2. **Overlap-Add method** for efficient FIR filtering
3. **Cross-correlation** for R-peak detection
4. **Otsu's thresholding** for automatic threshold selection
5. **Sobel/Prewitt edge detection** with gradient computation
6. **Wavelet denoising** using soft thresholding
7. **Histogram equalization** for contrast enhancement

## 🔮 Future Enhancements

- Real-time ECG processing with streaming data
- Advanced ML classifiers (CNN, LSTM, Transformer)
- Multi-lead ECG analysis (12-lead)
- Android/Web app integration
- Arrhythmia classification (AFib, VT, PVC, etc.)
- Beat-to-beat variability analysis
- QT interval measurement and analysis

## 👥 Contributors

- **Nicky John** - Project Lead and Developer

## 📄 License

This project is for educational purposes as part of DSP Mini-Project coursework.

## 🙏 Acknowledgments

- MIT-BIH Arrhythmia Database from PhysioNet
- Course instructors for guidance
- Open-source Python community

## 📞 Contact

For questions or issues, please open an issue on GitHub.

---

**Course**: Digital Signal Processing and Applications (DJS22CEC802 / DJS22CEL802)  
**Program**: B.Tech Computer Engineering  
**Academic Year**: 2024-2025
