# ECG Multi-Label Diagnosis with Deep 1D-CNNs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow / Keras](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Dataset: PTB-XL PhysioNet](https://img.shields.io/badge/Dataset-PTB--XL%20(21k%20records)-red.svg)](https://physionet.org/content/ptb-xl/1.0.3/)

> **Executive Summary (5-Minute Read):**  
> An end-to-end Deep Learning pipeline for automated multi-label diagnosis of cardiac pathologies from 12-lead raw ECG signals (PTB-XL dataset).  
> Solves diagnostic co-occurrence (e.g. Conduction Disturbance co-occurring with Myocardial Infarction) using specialized 1D-CNN architectures, rigorous patient-level stratification, and systematic signal noise/smoothing invariance testing across 100Hz and 500Hz sampling frequencies.

---

## 1. Clinical Challenge & Motivation

Standard automated ECG analysis frequently treats diagnostic categories as mutually exclusive, failing in real clinical settings where cardiac conditions regularly co-occur. Furthermore, baseline wander, high-frequency muscle tremor, and sampling rate differences significantly degrade off-the-shelf convolutional networks.

### Project Goals:
1. **Multi-Label Architecture:** Predict non-mutually exclusive diagnostic super-classes (CD: Conduction Disturbance, MI: Myocardial Infarction, STTC: ST/T Change, HYP: Hypertrophy, NORM: Normal ECG).
2. **Patient-Level Leakage Prevention:** Rigorous grouping and split assertions ensuring no patient has records in both training and test sets.
3. **Signal Robustness & Stress Testing:** Evaluate model resilience under Gaussian noise injection and moving-average smoothing at 100Hz and 500Hz sampling regimes.

---

## 2. Architecture & Deep Learning Pipeline

```mermaid
flowchart TD
    Raw["Raw 12-Lead ECG Records<br/>(10 seconds @ 100Hz / 500Hz)"] --> Pre["Signal Preprocessing<br/>(Zero-Padding, Baseline Correction)"]
    Pre --> Split["Patient-Stratified Split<br/>(Zero Cross-Patient Data Leakage)"]
    
    subgraph ModelArch ["1D-CNN Multi-Label Architecture"]
        Conv1["Conv1D + Batch Normalization + ReLU"]
        Pool1["MaxPooling1D / Spatial Dropout"]
        Conv2["Dilated Conv1D Residual Blocks"]
        GlobalPool["Global Average Pooling 1D"]
        DenseOut["Dense Output (Sigmoid Activation)<br/>Multi-Label Cross-Entropy Loss"]
    end
    
    subgraph StressTest ["Robustness & Invariance Suite"]
        Noise["Gaussian Noise Perturbations (σ = 0.01 - 0.05)"]
        Smooth["Moving Window Smoothing (Kernel = 10 - 50)"]
    end

    Split --> ModelArch
    ModelArch --> StressTest
    StressTest --> Eval["Clinical Evaluation Suite<br/>Macro/Micro ROC-AUC, Multi-Label Confusion Matrices"]
```

---

## 3. Experimental Findings & Methodological Rigor

| Experiment / Configuration | Sampling Rate | Macro ROC-AUC | Invariance / Clinical Takeaway |
| :--- | :--- | :--- | :--- |
| **Baseline Multi-Class** | 100Hz | 0.81 | Failed on overlapping conditions (e.g., CD + MI). |
| **Multi-Label 1D-CNN (Full PTB-XL)** | 100Hz | 0.88 | Accurately captured independent co-occurring pathologies. |
| **High-Resolution Regime** | 500Hz | **0.91** | High-frequency QRS morphological features improved CD detection. |
| **Noise Injection ($\sigma=0.03$)** | 500Hz | 0.89 | Maintained $>97\%$ relative performance; proved filter stability. |
| **Moving Average Smoothing** | 500Hz | 0.87 | Retained macro rhythm patterns despite mild attenuation of sharp R-peaks. |

### Key Methodological Audits:
- **Single-Sample vs Repeated Records:** Audited patient duplication effect; confirmed that patient-level deduplication maintains consistent calibration curves without artificial performance inflation.
- **Label Hierarchy Interaction:** Decomposed 27 fine-grained SCP-ECG statements into validated diagnostic super-classes.

---

## 4. Quickstart & Reproducibility

### Installation
```bash
# Clone repository
git clone https://github.com/RazMaoz/ecg-multi-label-cnn-diagnosis.git
cd ecg-multi-label-cnn-diagnosis

# Install dependencies
pip install -r requirements.txt
```

### Dataset Download
The dataset is publicly available on PhysioNet:
- [PTB-XL: A Large Publicly Available Electrocardiography Dataset](https://physionet.org/content/ptb-xl/1.0.3/)
- Place the downloaded `ptb-xl-1.0.3` folder or records in the `data/` directory (git-ignored automatically).

### Running the Exploration & Training Notebook
```bash
jupyter notebook notebooks/ecg_multi_label_cnn_pipeline.ipynb
```

---

## 5. Repository Structure

```
ecg-multi-label-cnn-diagnosis/
├── README.md                              # Executive clinical & technical overview
├── requirements.txt                       # Explicit dependency specifications
├── .gitignore                             # Filters for large data, checkpoints, and weights
├── notebooks/
│   └── ecg_multi_label_cnn_pipeline.ipynb # End-to-end multi-label training & stress testing
└── presentation/
    └── Naya_Final_Project_ECG_Diagnosis_with_CNN.pptx # Comprehensive presentation deck
```

---

## Author & Contact
**Raz Maoz**  
Data Scientist | Data Systems Architect  
- **LinkedIn:** [linkedin.com/in/raz-maoz](https://www.linkedin.com/in/raz-maoz/)  
- **GitHub:** [@RazMaoz](https://github.com/RazMaoz)  
- **Email:** `raz.m.maoz@gmail.com`
