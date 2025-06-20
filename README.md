
# 🛡️ UPI Transaction Fraud Detection System

A machine learning project to detect fraudulent UPI transactions using **XGBoost** and **Streamlit**, enhanced with class balancing via **SMOTE**. Users can input a single transaction or upload a CSV of multiple transactions to check for potential fraud in real-time.

---

## 📂 Project Structure

```
├── app.py                     # Streamlit app for fraud detection
├── balancing_data.py          # Script to balance the dataset using SMOTE
├── upi_data.csv               # Original transaction dataset
├── balanced_upi_data.csv      # Balanced dataset after SMOTE
├── UPI_Fraud_Detection.pkl    # Trained XGBoost model
├── encoder.pkl                # Trained OneHotEncoder
└── README.md                  # Project documentation
```

---

## 🚀 Features

- 📊 Predicts if a transaction is **fraudulent or legitimate**
- 🧠 Uses **XGBoost** classifier trained on balanced data
- 💡 Supports both **manual input** and **CSV upload**
- 🎛️ Encodes categorical fields with **One-Hot Encoding**
- 🔁 **SMOTE** used to handle class imbalance in training
- 🌐 Built with **Streamlit** for an interactive UI

---

## 📌 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

**Key Libraries:**
- `xgboost`
- `pandas`
- `scikit-learn`
- `streamlit`
- `imblearn`

---

## 📊 How to Use

### 🔹 1. Clone the repository

```bash
git clone https://github.com/your-username/upi-fraud-detector.git
cd upi-fraud-detector
```

### 🔹 2. Run the Streamlit app

```bash
streamlit run app.py
```

### 🔹 3. Interact

- Manually enter transaction details
- OR upload a `.csv` file with columns:  
  `Amount`, `Date`, `Transaction_Type`, `Payment_Gateway`, `Transaction_State`, `Merchant_Category`

---

## 🧠 Model Training

To balance your data and train the model:

```bash
python balancing_data.py
```

- This creates a `balanced_upi_data.csv`
- Use it to train the model with `XGBClassifier(scale_pos_weight=...)`

---

## 📎 Sample CSV Format

| Amount | Date       | Transaction_Type | Payment_Gateway | Transaction_State | Merchant_Category |
|--------|------------|------------------|------------------|-------------------|-------------------|
| 1499   | 2025-06-01 | Purchase         | Google Pay       | Tamil Nadu        | Purchases         |

---

## 🙌 Acknowledgments

- [Streamlit](https://streamlit.io/)
- [XGBoost](https://xgboost.ai/)
- [imbalanced-learn](https://imbalanced-learn.org/)
