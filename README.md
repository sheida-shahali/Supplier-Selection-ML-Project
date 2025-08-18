# 🧠 Supplier Prediction ML Components (Public Version)

This repository contains my contributions to a machine learning pipeline designed to predict the optimal **supplier** and **account** for each shipment order. The full system includes multiple modules; this public version shares only the components I directly developed.

---

## 🔍 Project Overview

The end-to-end ML system includes various classes and components, such as:

- 📦 **Shipment & Geolocation Handling** (not included)
- 🧭 **Location and Route Optimization** (not included)
- 🏷️ **Category Classification** ✅ *(developed by me)*
- 🔁 **Hierarchical Supplier Prediction** ✅ *(developed by me)*
- ⚙️ **Workflow Management & Integration** ✅ *(partially developed by me)*

---

## ✅ My Contributions

### 1. 🏷️ Category Classifier (`Category.py`)
An NLP pipeline that classifies orders based on the `goods_description` field.

- Text preprocessing (lemmatization, multilingual stopword removal)
- Embedding with **SBERT**
- Classification via **Random Forest**
- Keyword-based fallback rules
- Output: `final_category_code` (encoded)

---

### 2. 🔁 Hierarchical Supplier Prediction (`ModelManager.py`)
A two-level classifier using **XGBoost**:

- **Level 1:** Predicts supplier  
- **Level 2:** Predicts account based on supplier  
- Includes:
  - SMOTE for class balancing
  - Stratified cross-validation
  - Feature importance plots
  - Confusion matrix evaluation

---

### 3. ⚙️ Workflow Integration (`WorkflowManager.py`)
I implemented core functions to integrate the category classifier and supplier predictor:

- Training & evaluation management
- Model serialization and loading
- Input validation & preprocessing
- Note: Other classes like `Shipment` and `Location` were not developed by me

---

## ⚙️ Tech Stack

| Category      | Tools |
|---------------|-------|
| Language      | Python |
| ML Models     | XGBoost, Random Forest |
| NLP           | sentence-transformers (SBERT), SpaCy, NLTK |
| Evaluation    | scikit-learn, matplotlib |
| Data Handling | pandas, joblib |
| Sampling      | imbalanced-learn (SMOTE) |

---

## 🚫 Not Included

This repo **excludes** components I did not develop:

- Geolocation & shipment logic (`Shipment`, `Location`)
- Sensitive or proprietary datasets
- Pre-trained model binaries

---

## 🧪 Usage

> ⚠️ **Note:**  
> These modules were developed for a specific proprietary dataset with fixed columns and encodings.  
> As such, **they are not plug-and-play** and will require modification to work on other datasets.

#### You *can*:
- Review the logic for building hierarchical ML classifiers
- Learn how to integrate NLP and tabular models
- Adapt portions of the code for similar projects

#### You *cannot*:
- Directly run the code on new data without restructuring inputs
- Use the models without replicating the dataset schema

---

## 📄 License

This code is part of a **private project**, shared here only for educational or demonstrative purposes.  
**Do not use in production or commercial settings** without permission.
