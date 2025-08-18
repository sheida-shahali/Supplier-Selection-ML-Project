# Supplier Prediction ML Components (Public Version)

This repository contains my contributions to a larger machine learning system built for supplier and account prediction in a logistics context. The complete pipeline includes multiple modules; here, I've shared only the components I directly worked on.

---

## Project Overview

The overall system was designed to process shipment data and predict the most appropriate supplier and account combination based on order details.


### Components Included in This Repository (My Contributions)

- **Category Classification** – Classifies shipment orders based on text descriptions using NLP
- **Hierarchical Supplier and Account Prediction** – Predicts the supplier first, then the account within that supplier using a two-stage ML model
- **Workflow Integration** – Handles model training, evaluation, and integration into the broader pipeline

### Components Excluded (Not Developed by Me)

These modules are part of the full private system but not included in this public release:

- Shipment and geolocation processing
- Location and route optimization logic
- Full shipment modeling and business-specific features

---

## My Contributions

### 1. Category Classifier (`Category.py`)

A natural language processing pipeline that classifies shipment orders based on their `goods_description` field.

- Text cleaning and normalization (lemmatization, multilingual stopword removal)
- Embedding using **SBERT**
- Classification with **Random Forest**
- Keyword-based fallback logic
- Final output: `final_category_code` (encoded)

---

### 2. Hierarchical Supplier Prediction (`ModelManager.py`)

A two-level machine learning model implemented using **XGBoost**:

- **Step 1**: Predict the supplier from input features
- **Step 2**: Given the supplier, predict the corresponding account
- Built-in:
  - SMOTE resampling to address class imbalance
  - Stratified cross-validation
  - Confusion matrix visualization and feature importance analysis

---

### 3. Workflow Integration (`WorkflowManager.py`)

I developed the logic to integrate and manage both models within a single flow:

- Triggers for model training and evaluation
- Input preprocessing and format validation
- Saving and loading trained models

---

## Tech Stack

| Category      | Tools Used |
|---------------|------------|
| Programming   | Python |
| Machine Learning | XGBoost, Random Forest |
| NLP           | sentence-transformers (SBERT), SpaCy, NLTK |
| Data Handling | pandas, joblib |
| Evaluation    | scikit-learn, matplotlib |
| Resampling    | imbalanced-learn (SMOTE) |

---

## Preprocessed Dataset Structure 

The following is a simplified example of the preprocessed dataset used for model training. The raw data underwent multiple transformation steps (e.g., encoding, geolocation processing, feature engineering) and is not included due to confidentiality.

```text
| shipment_id | result_supplier_encoded | result_account_encoded | COD_NAZIONE_CLIFOR_MITT | CAP_CLIFOR_MITT | COD_NAZIONE_CLIFOR_DEST | CAP_CLIFOR_DEST | service_type_encoded | shipping_type_encoded | goods_type_encoded | is_dangerous | dogana | dry_ice | origin_lat | origin_long | dest_lat | dest_long | distance | total_weight | total_volume | dim1_max | dim2_max | dim3_max | number_of_parcels | Category | Category name               |
|-------------|-------------------------|------------------------|-------------------------|-----------------|-------------------------|-----------------|----------------------|-----------------------|--------------------|--------------|--------|---------|------------|-------------|----------|-----------|----------|--------------|--------------|----------|----------|----------|-------------------|----------|-----------------------------|
| 100001      | 4                       | 12                     | IT                      | 12345           | FR                      | 75001           | 1                    | 2                     | 0                  | 0            | 1      | 0       | 45.1234    | 9.5678      | 48.8566  | 2.3522    | 850.5    | 3.25         | 12000.0      | 40.0     | 35.0     | 25.0     | 1                 | 0        | Food and Beverages          |
| 100002      | 7                       | 9                      | DE                      | 10115           | ES                      | 28001           | 0                    | 1                     | 1                  | 0            | 0      | 0       | 52.5200    | 13.4050     | 40.4168  | -3.7038   | 1800.0   | 5.80         | 9500.0       | 38.0     | 32.0     | 20.0     | 2                 | 1        | Mechanical Spare Parts      |
```

### Column Name Reference

| Column Name                | Description                                      |
|---------------------------|--------------------------------------------------|
| `shipment_id`             | Unique shipment identifier                       |
| `result_supplier_encoded` | Encoded predicted supplier                       |
| `result_account_encoded`  | Encoded predicted account (within supplier)      |
| `COD_NAZIONE_CLIFOR_MITT` | Country code of origin customer                  |
| `CAP_CLIFOR_MITT`         | Postal code of origin customer                   |
| `COD_NAZIONE_CLIFOR_DEST` | Country code of destination customer             |
| `CAP_CLIFOR_DEST`         | Postal code of destination customer              |
| `service_type_encoded`    | Encoded service type                             |
| `shipping_type_encoded`   | Encoded shipping type                            |
| `goods_type_encoded`      | Encoded goods type                               |
| `is_dangerous`            | Dangerous goods flag                             |
| `dogana`                  | Customs required (boolean)                       |
| `dry_ice`                 | Contains dry ice (boolean)                       |
| `origin_lat`              | Origin latitude                                  |
| `origin_long`             | Origin longitude                                 |
| `dest_lat`                | Destination latitude                             |
| `dest_long`               | Destination longitude                            |
| `distance`                | Distance in kilometers                           |
| `total_weight`            | Total weight of shipment                         |
| `total_volume`            | Total volume of shipment                         |
| `dim1_max`, `dim2_max`, `dim3_max` | Max parcel dimensions in cm          |
| `number_of_parcels`       | Number of parcels in the shipment                |
| `Category`                | Encoded predicted category                       |
| `Category name`           | Human-readable predicted category name           |

> Note: All values are simulated for illustration only.


### Column Descriptions:
- **Target Labels**:  
  - `result_supplier_encoded`, `result_account_encoded`: used in hierarchical ML model  
  - `Category`, `Category name`: output from NLP category classifier  

- **Features**:  
  - Location and postal code fields (`CAP_`, `COD_NAZIONE_`, lat/long)  
  - Shipping and goods metadata (types, weight, volume, parcel dimensions)  
  - Flags for special handling (`is_dangerous`, `dogana`, `dry_ice`)  

---

## Usage

> ⚠️ These modules were tailored to a proprietary dataset and schema. They are **not plug-and-play** and require adaptation for other use cases.

You can:
- Review logic for hierarchical ML modeling
- Explore NLP integration in tabular data pipelines
- Reuse preprocessing, model training, and evaluation code

You cannot:
- Directly apply the code to new datasets without restructuring
- Run pre-trained models (models are not included)

---

## License

This repository contains selected parts of a private, real-world project.  
Provided solely for educational and demonstrative purposes.  
**Not licensed for production or commercial use.**
