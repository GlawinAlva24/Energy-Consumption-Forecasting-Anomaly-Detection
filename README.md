# ⚡ Energy Consumption Forecasting & Anomaly Detection

**MSc Dissertation – Artificial Intelligence with Business Strategy**  
*By Glawin John Alva – Supervised by Dr. Farzaneh Farhadi*

This project explores hybrid deep learning models to forecast household electricity consumption and detect anomalies that may signify device faults, inefficiencies, or abnormal usage behavior.

---

## 🚀 Project Highlights

- 🔀 **Hybrid Forecasting**: LSTM + Stationary Wavelet Transform (SWT) improves accuracy by filtering noise
- 🔍 **Anomaly Detection**: Isolation Forest, Autoencoders, One-Class SVM, and Local Outlier Factor
- 📊 **Datasets Used**:
  - UCI Household Electric Power Consumption (Forecasting)
  - LEAD1.0 (Labelled Anomaly Detection – Commercial Buildings)
- 📈 **Key Metrics**: RMSE, MAE, F1-score, Precision, Recall

---

## 📊 Forecasting Results

| Model               | Dataset      | RMSE   | MAE    |
|--------------------|--------------|--------|--------|
| SWT-LSTM (Hybrid)  | UCI          | 0.0220 | 0.0125 |
| SWT-LSTM (Hybrid)  | LEAD1.0      | 0.0386 | 0.0284 |
| CNN-LSTM           | UCI          | 0.0584 | 0.0418 |

---

## 🔍 Anomaly Detection Results (LEAD1.0)

| Method            | Precision | Recall | F1-score |
|------------------|-----------|--------|----------|
| Isolation Forest |     –     |   –    |     –    |
| Autoencoder      |     –     |   –    |     –    |
| One-Class SVM    |     –     |   –    |     –    |
| LOF              |     –     |   –    |     –    |

*Values can be updated from your final tables*

---

## 📚 Methodology

**Hybrid Forecasting Pipeline**:
1. Preprocess time-series data (resample, normalize, lag features)
2. Apply Stationary Wavelet Transform (SWT)
3. Train LSTM on approximation coefficients
4. Predict future hourly consumption

**Anomaly Detection Pipeline**:
- Use residuals (actual - predicted) for unsupervised detection
- Apply:
  - Isolation Forest (spikes)
  - Autoencoders (reconstruction error)
  - One-Class SVM (subtle deviations)
  - LOF (local density outliers)

---

## 🧠 Technologies Used

- Python, Jupyter Notebooks
- Pandas, NumPy, Matplotlib, Seaborn
- TensorFlow / Keras
- Scikit-learn, PyWavelets
- SMOTE (imbalanced anomaly data)
- Google Colab, Visual Studio Code

---

## 🛠️ Project Structure

