
#  House Price Prediction - ANN Experimental Analysis

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Status](https://img.shields.io/badge/Academic-Project-brightgreen.svg)

This repository contains an experimental study focused on predicting real estate prices using **Artificial Neural Networks (ANN)**. The project involved testing **16 different architectures** to find the optimal configuration for regression.

---

## 📋 Project Overview
The goal was to analyze how different ANN structures (varying layers and neurons) impact prediction accuracy.
*   **Dataset**: Housing Price Dataset (546 records, 11 features).
*   **Task**: Regression (Predicting the `price` variable).
*   **Experimental Approach**: Comparative analysis of 16 models (RNA 1 to RNA 16).

## 🔬 Methodology & Preprocessing
To ensure high-quality training, the following pipeline was implemented:
*   **Data Cleaning**: Categorical "yes/no" mapping to binary values.
*   **One-Hot Encoding**: Applied to `furnishingstatus` to avoid numerical bias.
*   **Feature Scaling**: Used `StandardScaler` to normalize data (Mean=0, StdDev=1).
*   **Validation Strategy**: **5-Fold Cross-Validation** was used across all experiments to ensure result stability.
*   **Optimization**: Adam optimizer with `EarlyStopping` (patience=8) to prevent overfitting.

## 📂 Project Structure
```text
├── data/ housing.csv               # Original dataset (546 entries)
├── environment.yml           # Anaconda environment (laboratoareIA)
├── house_price_prediction.ipynb # Experimental Notebook
├── README.md                 # Project documentation
└── docs/ Documentation.pdf    # Project documentation, for more details of the experiment
```

---

## 📊 Experimental Results
After testing 16 different configurations, **RNA 16** was identified as the most efficient architecture.

### 🏆 The Winning Architecture (RNA 16)
| Parameter | Configuration |
| :--- | :--- |
| **Hidden Layers** | 4 Layers (15, 30, 12, 40 neurons) |
| **Activation** | ReLU |
| **Training Epochs** | Max 500 (Early Stopping triggered) |
| **Final RMSE** | 1,104,816.30 |
| **Final Loss (MSE)** | 0.1724 |

*The study showed that increasing the depth of the network (4 layers) while maintaining a balanced number of neurons significantly reduced the prediction error.*

---

## 🚀 Installation & Setup

### 1️⃣ Clone & Navigate
```bash
git clone https://github.com/andrei-vasile-dev/house-price-prediction-nn.git
cd house-price-prediction-nn
```

### 2️⃣ Environment Setup
Create and activate the specialized environment:
```bash
conda env create -f environment.yml
conda activate laboratoareIA
```

### 3️⃣ Jupyter Integration
Register the kernel to run the notebook:
```bash
pip install ipykernel
python -m ipykernel install --user --name laboratoareIA --display-name "Python (AI-Housing)"
```

### 4️⃣ Run Experiments
```bash
jupyter notebook "house_price_prediction.ipynb"
```
> **Note:** Remember to select the **"Python (AI-Housing)"** kernel from the Jupyter menu.

---
👤 **Author**: Vasile Andrei – Daniel  
🎓 **Course**: Artificial Intelligence (Computer Science, Year III)  
⭐ *Academic project focused on Neural Network optimization.*