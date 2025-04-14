# ⚡ Energy Consumption Forecasting & Anomaly Detection

**MSc Dissertation – Artificial Intelligence with Business Strategy**  
*By Glawin John Alva – Supervised by Dr. Farzaneh Farhadi*

This project applies deep learning and hybrid models to forecast household energy consumption and detect anomalies using both statistical and AI-based techniques.

---

## 🚀 Highlights

- 🔀 **Hybrid Forecasting**: LSTM + Stationary Wavelet Transform (SWT) for denoised, accurate predictions
- 🔍 **Unsupervised Anomaly Detection**: Autoencoder, Isolation Forest, LOF, and One-Class SVM
- ⚖️ **Imbalanced Data Handling**: SMOTE applied to boost anomaly training samples
- 📦 **Datasets Used**:
  - UCI Household Power Consumption Dataset
  - LEAD1.0 Dataset (Labelled commercial anomalies)

---

## 🗃️ Folder Structure
Energy-Consumption-Forecasting-Anomaly-Detection/ 
│ 
├── database/ 
│   ├── UCI_dataset.zip 
│   ├── LEAD1_dataset.zip   
│   └── household_power_consumption.csv  
│
├── data/  
│   └── README.md  
│
├── notebooks/  
│   ├── forecasting/  
│   │   ├── CNN_AND_LSTM_MODEL.ipynb  
│   │   └── Hybrid_UCI_dataset.ipynb  
│   ├── anomaly_detection/  
│   │   └── final1_Lead-with_SMOTE.ipynb  
│
├── models/
│   └── best_model.keras
│
├── results/
│   ├── anomaly_detection_results_summary.xlsx
│   └── output_graphs/
│       ├── swt_decomposition.png
│       ├── residuals_plot.png
│       └── lof_anomalies.png
│
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE

---

## 📈 Results Summary

### Forecasting (RMSE / MAE)

| Model               | Dataset      | RMSE   | MAE    |
|--------------------|--------------|--------|--------|
| SWT-LSTM (Hybrid)  | UCI          | 0.0220 | 0.0125 |
| SWT-LSTM (Hybrid)  | LEAD1.0      | 0.0386 | 0.0284 |
| CNN-LSTM           | UCI          | 0.0584 | 0.0418 |

### Anomaly Detection (on LEAD1.0)

| Method            | Precision | Recall | F1-score |
|------------------|-----------|--------|----------|
| Isolation Forest | 0.72      | 0.68   | 0.70     |
| Autoencoder      | 0.78      | 0.74   | 0.76     |
| One-Class SVM    | 0.60      | 0.52   | 0.56     |
| LOF              | 0.67      | 0.58   | 0.62     |

---

## 💾 Usage

### 1. Clone the Repository

```bash
git clone https://github.com/GlawinAlva24/Energy-Consumption-Forecasting-Anomaly-Detection.git
cd Energy-Consumption-Forecasting-Anomaly-Detection

---


### 2. Unzip Datasets

Use any archive tool like **7-Zip** or **WinRAR** to extract:

- `UCI_dataset.zip`
- `LEAD1_dataset.zip`

Place the contents in the appropriate `data/` subfolders.

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run Notebooks

You can start Jupyter Notebook and run any of the following:

- 📊 Forecasting: `notebooks/forecasting/Hybrid_UCI_dataset.ipynb`
- 🤖 Anomaly Detection: `notebooks/anomaly_detection/final1_Lead-with_SMOTE.ipynb`
- 📈 CNN vs LSTM: `notebooks/forecasting/CNN_AND_LSTM_MODEL.ipynb`

---

## 🛠 Technologies

- Python (Jupyter Notebooks)
- TensorFlow / Keras
- PyWavelets
- Scikit-learn, Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn
- NumPy, Pandas

---

## 🧠 Future Work

- Real-time streaming integration (Kafka or MQTT)
- Streamlit dashboard for anomaly feedback
- Expand to multi-household or grid-level predictions
- Live IoT integration and alerts

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

- Dr. Farzaneh Farhadi (Supervisor)
- University of Aston University
- Dataset contributors: UCI Machine Learning Repository & LEAD1.0
