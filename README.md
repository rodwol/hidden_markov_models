# Hidden Markov Models for Activity Recognition

This Jupyter notebook uses **Gaussian Hidden Markov Models (HMM)** to classify human activities from smartphone accelerometer data. The pipeline covers data loading, preprocessing, feature extraction, and HMM-based classification.

## Overview

The notebook trains a Gaussian HMM to recognize four activities—**still**, **standing**, **walking**, and **jumping**—from 3-axis accelerometer readings. HMMs are used because they model temporal structure and state transitions, which fits activity recognition well.

## Data

- **Source**: Kaggle dataset [`rodasgoniche/data-new`](https://www.kaggle.com/datasets/rodasgoniche/data-new)
- **Sensor**: Total acceleration (x, y, z)
- **Activities**: still, standing, walking, jumping (~29k samples each)
- **Format**: CSV files organized by session and activity

## Pipeline

1. **Data loading** — Downloads the dataset via `kagglehub` and loads CSV files from session folders.
2. **Preprocessing** — Removes duplicates, sorts by time, resamples to 50 Hz, and computes acceleration magnitude.
3. **Filtering** — Applies a 4th-order Butterworth low-pass filter (10 Hz cutoff) to reduce noise.
4. **Windowing** — 2-second windows (100 samples) with 1-second step; windows with &lt;80% activity agreement are dropped.
5. **Feature extraction** — For each axis (x, y, z, acc_mag): mean, std, max, min, range, median, IQR, RMS, SMA, dominant frequency, and spectral energy (44 features total).
6. **Train/test split** — 80/20 session-based split per activity; features standardized with `StandardScaler`.
7. **Model** — `GaussianHMM` from `hmmlearn` with 4 states (one per activity), full covariance, and 1000 EM iterations.
8. **Evaluation** — Accuracy, confusion matrix, classification report, transition matrix, and emission analysis.

## Requirements

- `hmmlearn`
- `numpy`, `pandas`, `matplotlib`
- `scipy` (signal processing)
- `scikit-learn` (preprocessing, metrics)
- `kagglehub` (dataset download)

## Setup

1. Install dependencies: `pip install hmmlearn kagglehub`
2. Configure Kaggle credentials if needed for dataset access.
3. Open `Hidden_Markov_Models.ipynb` and run all cells.

The notebook can also be run in [Google Colab](https://colab.research.google.com/github/rodwol/hidden_markov_models/blob/main/Hidden_Markov_Models.ipynb).
