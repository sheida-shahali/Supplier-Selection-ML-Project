"""
WorkflowManager (Public Version)

Handles the end-to-end pipeline orchestration for category classification and hierarchical supplier prediction.
External dependencies (like Shipment/Location) are excluded or referenced as placeholders.
"""

import pandas as pd
import numpy as np
import logging
import json
from sklearn.preprocessing import LabelEncoder


from ModelManager import ModelManager
# from Location import Location  # External
# from Shipment import Shipment  # External
from Category import Category



class WorkflowManager:
    """
    Manages the overall workflow, including data processing, geocoding,
    shipment creation, model training, and evaluation.
    """


    def __init__(self, data_filepath, output_filepath, cleaned_dataset,
                 encoder_path, model_save_path, labeled_data_path,
                 geocode_cache_csv, unresolved_csv, manual_fixes_csv):
        self.data_filepath = data_filepath
        self.output_filepath = output_filepath
        self.cleaned_dataset = cleaned_dataset
        self.encoder_path = encoder_path
        self.labeled_data_path = labeled_data_path
        self.category_model = Category()
        self.model_manager = ModelManager(mode=model_type)
        self.model_type = model_type
        self.model_save_path = model_save_path
        self.geocode_cache_csv = geocode_cache_csv
        self.manual_fixes_csv  = manual_fixes_csv
        self.unresolved_csv = unresolved_csv
        self.shipments = []
        self.data = None


    def load_data(self):
        """Loads Excel data into a pandas DataFrame."""
        logging.info("Loading data...")
        cols_to_use = ["PRENOTAZIONE", "Result_Supplier", "Result_Account", "goods_type",
                      "service_type", "shipping_type", "COD_NAZIONE_CLIFOR_MITT",
                      "CAP_CLIFOR_MITT","COD_NAZIONE_CLIFOR_DEST", "CAP_CLIFOR_DEST",
                      "is_dangerous", "dogana", "DryIce", "colli", "goods_description"]
        self.data = pd.read_excel(self.data_filepath, usecols=cols_to_use, keep_default_na=False, engine='openpyxl') # Ensures "NA" is read as a string
        #self.data = pd.read_csv(self.cleaned_dataset, usecols=cols_to_use, na_values=[], keep_default_na=False)

        # Drop rows where 'colli' is missing (i.e., NaN) immediately after loading
        self.data.dropna(subset=['colli'], inplace=True)


    def manage_rows(self):
        #Define supplier rows to keep
        rows_to_keep = [
            ("dhl", "10630****"),
            ("fedex", "21817****"),
            ("dhl", "95698****"),
            ("tnt", "886****"),
            ("gls", "****"),
            ("bet","***"),
            ("dhl","95698****"),
            ("tnt","688****"),
            ("fedex","43662****"),
            ("fedex","***")
        ]
        self.data = self.data[self.data.set_index(['Result_Supplier', 'Result_Account']).index.isin(rows_to_keep)]
        self.data = self.data.reset_index(drop=True)
        # Save the cleaned dataset
        self.data.to_csv(self.cleaned_dataset, index=False)
        logging.info(f"Filtered dataset saved: {cleaned_dataset}")

        
    def clean_string_columns(self):
        """Strips whitespace from all relevant string columns before encoding."""
        columns_to_clean = [
            "Result_Supplier",
            "Result_Account",
            "goods_type",
            "service_type",
            "shipping_type"
        ]
        for col in columns_to_clean:
            self.data[col] = self.data[col].astype(str).str.strip()


    def preprocess_data(self):
        """Clean and preprocess the DataFrame (drop duplicates, handle missing values, etc.)."""
        logging.info("Preprocessing data...")
        # Delete duplication of rows
        self.data.drop_duplicates(subset=['PRENOTAZIONE'], keep='first', inplace=True)

        # Standardize Names
        self.data['Result_Supplier'] = self.data['Result_Supplier'].str.lower().str.strip()
        self.data['Result_Account'] = self.data['Result_Account'].astype(str)


        # Replacement rows
        account_replacements = {"202506542": "218174280"}
        self.data['Result_Account'] = self.data['Result_Account'].replace(account_replacements)

        # Convert numerical columns to integers
        self.data['dogana'] = pd.to_numeric(self.data['dogana'], errors='coerce').fillna(0).astype(int)
        self.data['DryIce'] = pd.to_numeric(self.data['DryIce'], errors='coerce').fillna(0).astype(int)
        self.data["PRENOTAZIONE"] = self.data["PRENOTAZIONE"].astype(str)

        # Handle missing values
        for column in self.data.columns:
            if self.data[column].dtype == "object" or self.data[column].dtype == "string" or self.data[column].isna().any():
                # Ensure the column is treated as an object if it contains NaN
                self.data[column] = self.data[column].astype(str).fillna("UNKNOWN")
            elif np.issubdtype(self.data[column].dtype, np.number):  # For numerical columns
                self.data[column] = self.data[column].fillna(0)


    def encode_data(self):
        """Encodes categorical variables, preserves mappings, and stores encoders."""
        logging.info("Encoding data and preserving mappings...")
        self.encoders = {}
        self.data["Supplier_Account"] = self.data["Result_Supplier"] + "_" + self.data["Result_Account"]

        # Columns to encode
        columns_to_encode = {
            "Supplier_Account": "supplier_account_encoded",
            "Result_Supplier": "supplier_encoded",
            "Result_Account": "account_encoded",
            "goods_type": "goods_type_encoded",
            "service_type": "service_type_encoded",
            "shipping_type": "shipping_type_encoded"
        }

        # Apply label encoding and store encoders
        for original_col, encoded_col in columns_to_encode.items():
            encoder = LabelEncoder()
            self.data[encoded_col] = encoder.fit_transform(self.data[original_col])
            self.encoders[encoded_col] = encoder

        # Save encoder class mappings to JSON
        encoders_json = {
            name: list(encoder.classes_)
            for name, encoder in self.encoders.items()
        }

        with open(self.encoder_path, "w") as f:
            json.dump(encoders_json, f, indent=2)

        logging.info("Encoding complete. Encoders saved successfully.")



    #def geocode_and_merge(self):
        """Extracts unique geocode combos, processes geocoding, and merges results into self.data."""
        """Load caches, geocode new combos, write unresolved list, merge lat/lon."""



    def train_category_classifier(self):
        logging.info("Training and applying category model in batch...")
        model = Category()
        labeled_data = pd.read_csv(self.labeled_data_path)
        model.train_classifier(labeled_data)
        self.data = model.categorize_data(self.data)



    #def create_shipments(self):
        

    def impute_numeric_nans(self, fill_value: float = 0.0):
        """
        Replace any remaining NaN values in *numeric* columns with
        ``fill_value`` (default 0.0).
        """
        numeric_cols = self.data.select_dtypes(include=["number"]).columns
        self.data[numeric_cols] = self.data[numeric_cols].fillna(fill_value)
        return self


    #def save_shipments_to_csv(self):



    def run_full_workflow(self):
        try:
            logging.info("Starting workflow...")
            self.load_data()
            self.preprocess_data()
            self.manage_rows()
            self.clean_string_columns()
            self.encode_data()
            #self.geocode_and_merge()
            self.train_category_classifier()
            #self.create_shipments()
            self.impute_numeric_nans()
            #self.save_shipments_to_csv()

            logging.info("Running model training and evaluation...")
            self.model_manager.run_training(csv_path=self.output_filepath, encoder_path=self.encoder_path,
                                            model_save_path=self.model_save_path)
            self.model_manager.run_evaluation(csv_path=self.output_filepath, encoder_path=self.encoder_path,
                                              model_path=self.model_save_path)

            logging.info("Workflow completed successfully.")

        except Exception as e:
            logging.error(f"Workflow failed: {e}", exc_info=True)

 # Example usage
if __name__ == "__main__":
    data_filepath = '/content/drive/...'
    output_filepath = '/content/drive/...'
    cleaned_dataset = '/content/drive/...'
    encoder_path = '/content/drive/...'
    model_save_path = '/content/drive/...'
    labeled_data_path = '/content/drive/...'
    geocode_cache_csv = '/content/drive/...'
    manual_fixes_csv = '/content/drive/...'
    unresolved_csv = '/content/drive/...'
    model_type = 'hierarchical'

    workflow = WorkflowManager(data_filepath, output_filepath,
                               cleaned_dataset, encoder_path,
                               model_save_path,
                               labeled_data_path, geocode_cache_csv,
                               unresolved_csv, manual_fixes_csv)
    workflow.run_full_workflow()
    logging.info("Workflow finished.")
