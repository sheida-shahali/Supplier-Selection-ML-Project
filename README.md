# Supplier Prediction ML Components (Public Version)

This repository contains my contributions to a machine learning pipeline designed to predict the optimal supplier and account for each shipment order. The complete system includes multiple components; this public version shares only the parts I directly developed.

---

## 🔍 Overview

My work focused on two main components:

### 1. 🏷️ Category Classifier
An NLP-based model that classifies shipment orders based on the `goods_description` field.

- **Text Preprocessing**: Lemmatization, stopword removal (EN & IT), normalization
- **Model**: SBERT embeddings + Random Forest
- **Fallback Rules**: Keyword-based subcategory mapping
- **Output**: Encoded `final_category_code`

### 2. 🔁 Hierarchical Supplier Prediction
A two-level machine learning model using XGBoost:

- **Level 1**: Predicts the supplier
- **Level 2**: Predicts the account within the predicted supplier
- **Features**:
  - SMOTE for class imbalance
  - Stratified cross-validation
  - Feature importance visualization
  - Confusion matrix evaluation

---

## 🧠 My Contributions

- Implemented the entire `Category` classification module
- Developed the hierarchical classification logic in `ModelManager`
- Integrated both models into `WorkflowManager`
- Added support for cross-validation, evaluation, and model persistence

---

## ⚙️ Tech Stack

- **Languages**: Python
- **ML**: XGBoost, Random Forest, StratifiedKFold
- **NLP**: `sentence-transformers` (SBERT), SpaCy, NLTK
- **Utilities**: pandas, sklearn, imbalanced-learn, joblib

---

## 🚫 Not Included

This repository does not include:
- Full pipeline components (e.g., geolocation, shipment modeling)
- Full dataset or proprietary data
- Pre-trained models (excluded for privacy)

---

## 🧪 How to Use

You may:
- Train the category classifier using labeled descriptions
- Run the hierarchical supplier model on encoded datasets
- Evaluate model performance with provided utilities

---

## 📄 License

This is a personal contribution extracted from a larger private project. Use of this code is for educational or demonstrative purposes only.
