# 🏜️ KrackHack 3.0 — Off-road Semantic Scene Segmentation (AI/ML Track)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![CUDA](https://img.shields.io/badge/CUDA-12.x-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Active-success)

A reproducible semantic segmentation pipeline for **autonomous off-road navigation** in desert environments, developed for **KrackHack 3.0 (AI/ML Track)**.
---

## 📌 Table of Contents

* [Project Overview](#project-overview)
* [Repository Structure](#repository-structure)
* [Dataset Setup](#dataset-setup)
* [Environment Setup](#environment-setup)
* [How to Run](#how-to-run)

  * [Training & Validation](#training--validation)
  * [Testing / Evaluation](#testing--evaluation)
* [Pretrained Models](#pretrained-models)
* [Reports & Documentation](#reports--documentation)
* [Tech Stack](#tech-stack)
* [Team](#team)
* [Notes on Reproducibility](#notes-on-reproducibility)

---

## 🧠 Project Overview

The objective of this project is to build a **robust multiclass semantic segmentation model** for **off-road desert scenes**, capable of generalizing from **synthetic training data** to **unseen test environments**.

The development followed a failure-driven progression:

* CNN baselines (DeepLabV3+)
* Aggressive domain randomization
* Stabilization via SWA
* Transformer-based SegFormer architecture

All experiments, results, and reasoning are documented in the reports included in this repository.

---

## 📁 Repository Structure

```
.
├── pipeline.ipynb              # Main training / validation / testing pipeline
├── run.txt                     # CUDA + PyTorch install commands (recommended)
├── training_curves.png         # Training & validation plots
├── inference_results.png       # Qualitative predictions
├── mid_submission_report.pdf   # Mid submission report
├── final_report.pdf            # Final report
├── requirements.txt
├── dataset/                    # Training + validation dataset 
│   ├── train/
│   └── val/
├── testing_dataset/            # Blind test dataset 
│   ├── Color_Images/
│   └── Segmentation/
└── README.md
```

---

## 📦 Dataset Setup

### Training & Validation Dataset

Download and extract **into `/dataset`**:

🔗
[https://storage.googleapis.com/duality-public-share/Hackathons/Duality%20Hackathon/Offroad_Segmentation_Training_Dataset.zip](https://storage.googleapis.com/duality-public-share/Hackathons/Duality%20Hackathon/Offroad_Segmentation_Training_Dataset.zip)

Expected structure:

```
dataset/
├── train/
│   ├── Color_Images/
│   └── Segmentation/
└── val/
    ├── Color_Images/
    └── Segmentation/
```

---

### Testing Dataset

Download and extract **into `/testing_dataset`**:

🔗
[https://storage.googleapis.com/duality-public-share/Hackathons/Duality%20Hackathon/Offroad_Segmentation_Training_Dataset.zip](https://storage.googleapis.com/duality-public-share/Hackathons/Duality%20Hackathon/Offroad_Segmentation_Training_Dataset.zip)

Expected structure:

```
testing_dataset/
├── Color_Images/
└── Segmentation/
```

---

## 🛠️ Environment Setup

### 1. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

---

### 2. Install Base Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Install CUDA-Compatible PyTorch (IMPORTANT)

After installing `requirements.txt`, **run the commands in `run.txt`**:

```bash
# Example
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> ⚠️ **Recommended:**
> `run.txt` contains the exact PyTorch + CUDA installation used during development.
> Running it ensures GPU compatibility and avoids common CUDA mismatches.

---

## ▶️ How to Run

### Training & Validation

1. Open `pipeline.ipynb`
2. Run **cells sequentially from top to bottom**
3. The notebook handles:

   * Dataset loading
   * Augmentations
   * Model initialization
   * Training + validation loops
   * Metric logging (mIoU)
   * Checkpoint saving

All reported results were generated using this pipeline.

---

### Testing / Evaluation

To evaluate a trained model on the **blind test dataset**:

1. Open `pipeline.ipynb`
2. Run:

   * Import cells
   * Dataset definitions
   * Model loading cell
   * **Testing / evaluation cells only**

This computes:

* Per-class IoU
* Mean IoU
* Optional qualitative visualizations

---

## 🧩 Pretrained Models

Final trained checkpoints are available here:

🔗
[https://drive.google.com/drive/folders/1cNMqY7EgFQZIi8m4-r8dFjYLib9pa6Nn](https://drive.google.com/drive/folders/1cNMqY7EgFQZIi8m4-r8dFjYLib9pa6Nn)

Use these weights for direct evaluation or inference without retraining.

---

## 📄 Reports & Documentation

All technical details, architectural decisions, experiments, and analyses are fully documented in:

* 📘 **Mid Submission Report**
  `mid_submission_krackhack.pdf`

* 📕 **Final Report**
  `final_report.pdf`

The README focuses on **usage and reproducibility** — refer to the reports for methodology.

---

## ⚙️ Tech Stack

* **Language:** Python
* **Framework:** PyTorch
* **Models:** DeepLabV3+, SegFormer
* **Libraries:**

  * segmentation-models-pytorch
  * Albumentations
  * OpenCV
  * NumPy
* **Hardware:** NVIDIA RTX 4060 (8GB)
* **Training:** Mixed Precision (AMP), AdamW, Cosine Scheduling

---

## 👥 Team

* **Aryan Sisodiya (Team Leader)**
  [InfinityXr9](https://github.com/InfinityXr9)

* **Daksh Rathi**
  [dakshrathi-india](https://github.com/dakshrathi-india)

* **Farhan Alam**
  [Frozen-afk](https://github.com/Frozen-afk)

* **Tanmay Pratap Singh**
  [DhmalTPS](https://github.com/DhmalTPS)

---

## ♻️ Notes on Reproducibility

* Fixed random seeds where applicable
* No test-set leakage during training
* Validation used strictly for model selection
* Blind test dataset evaluated only post-training
* Checkpoints, plots, and metrics align with the reports

If something does **not reproduce**, it is a bug — please open an issue.

---
<p style="text-align:center;">
Made with ❤️ by <b>Aryan Sisodiya</b>, <b>Daksh Rathi, <b>Farhan Alam</b>, and <b>Tanmay Pratap Singh</b> for <b>KrackHack 3.0</b>
</p>
