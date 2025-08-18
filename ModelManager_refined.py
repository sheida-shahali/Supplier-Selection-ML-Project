"""
ModelManager (Public Version)

This class implements hierarchical and ensemble classification using XGBoost.
Sensitive paths and proprietary references have been removed or replaced.
Author: [Your Name]
"""


import pandas as pd
import joblib
import json
import json
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    recall_score, f1_score, ConfusionMatrixDisplay
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import logging


class ModelManager:

    def __init__(self, mode='ensemble', n_splits=5, random_state=42, class_names=None):
        self.mode = mode
        self.n_splits = n_splits
        self.random_state = random_state
        self.class_names = class_names

                self.feature_importances_ = None
        self.feature_names = None

        if self.mode == 'hierarchical':
            self.supplier_model = None
            self.supplier_encoder = LabelEncoder()
            self.account_models = {}

    """Train the hierarchical model with supplier and account levels."""

    def run_training(self, csv_path, encoder_path, model_save_path):
        df = pd.read_csv(csv_path)
        X = df.drop(columns=[
            "supplier_account_encoded", "shipment_id", "supplier_encoded", "account_encoded"
        ]).values

        y_encoded = df["supplier_account_encoded"].values
        y = df[["supplier_encoded", "account_encoded"]].values if self.mode == 'hierarchical' else y_encoded

        self.feature_names = df.drop(columns=[
            "supplier_account_encoded", "shipment_id", "supplier_encoded", "account_encoded"
        ]).columns.tolist()

        with open(encoder_path, "r") as f:
            encoder_classes = json.load(f)

        # Rebuild all encoders
        encoders = {}
        for name, classes in encoder_classes.items():
            le = LabelEncoder()
            le.classes_ = np.array(classes)
            encoders[name] = le
        target_encoder = encoders["supplier_account_encoded"]
        self.class_names = target_encoder.classes_.tolist()

        stratify_labels = y[:, 0] if self.mode == 'hierarchical' else y
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        self.cv_metrics = {'supplier_f1': [], 'account_f1': [], 'joint_acc': []} if self.mode == 'hierarchical' else {
            'accuracy': [], 'recall': [], 'f1': [], 'all_true': [], 'all_pred': []
        }

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, stratify_labels)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            if self.mode == 'hierarchical':
                self._train_and_eval_hierarchical(X_train, y_train, X_val, y_val, fold)
            
        
        self.save_model(model_save_path)

    """Evaluate a trained hierarchical model on a given dataset."""

    def run_evaluation(self, csv_path, encoder_path, model_path):
        df = pd.read_csv(csv_path)
        X = df.drop(columns=[
            "supplier_account_encoded", "shipment_id", "supplier_encoded", "account_encoded"
        ]).values

        y_encoded = df["supplier_account_encoded"].values
        y = df[["supplier_encoded", "account_encoded"]].values if self.mode == 'hierarchical' else y_encoded

        model = ModelManager.load_model(model_path)
        model.feature_names = self.feature_names
        model.class_names = self.class_names

        if model.mode == 'hierarchical':
            supplier_f1, account_f1, joint = model._evaluate_hierarchical_fold(X, y)
            print(f"Supplier F1: {supplier_f1:.4f}")
            print(f"Account F1: {account_f1:.4f}")
            print(f"Joint Accuracy: {joint:.4f}")
            
            with open(encoder_path, "r") as f:
                encoder_classes = json.load(f)

            with open(encoder_path, "r") as f:
                encoder_classes = json.load(f)

            # Rebuild all encoders
            encoders = {}
            for name, classes in encoder_classes.items():
                le = LabelEncoder()
                le.classes_ = np.array(classes)
                encoders[name] = le
            supplier_encoder = encoders["supplier_encoded"]
            account_encoder = encoders["account_encoded"]

            predictions = model._predict_hierarchical(X)
            y_true_sup = supplier_encoder.inverse_transform(y[:, 0])
            y_true_acc = account_encoder.inverse_transform(y[:, 1])
            y_pred_sup = supplier_encoder.inverse_transform([p[0] for p in predictions])
            y_pred_acc = account_encoder.inverse_transform([p[1] for p in predictions])

                        
                        
                                    
            model._evaluate_combined_classification(
                list(zip(y_true_sup, y_true_acc)),
                list(zip(y_pred_sup, y_pred_acc))
            )
            model._plot_combined_confusion_matrix(
                list(zip(y_true_sup, y_true_acc)),
                list(zip(y_pred_sup, y_pred_acc)),
                title="Combined Supplier-Account Confusion Matrix"
            )
            model._plot_feature_importance()def _train_and_eval_hierarchical(self, X_train, y_train, X_val, y_val, fold):
        suppliers, accounts = y_train[:, 0], y_train[:, 1]
        y_sup_encoded = self.supplier_encoder.fit_transform(suppliers)

        # Train supplier model
        X_res, y_sup_res = SMOTE(random_state=self.random_state).fit_resample(X_train, y_sup_encoded)
        counts = Counter(y_sup_res)
        sample_weight = np.array([max(counts.values()) / counts[label] for label in y_sup_res])

        self.supplier_model = XGBClassifier(
            learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
            eval_metric='mlogloss', random_state=self.random_state, n_estimators=350
        )
        self.supplier_model.fit(X_res, y_sup_res, sample_weight=sample_weight)
        self._store_feature_importances(self.supplier_model)

        # Train per-supplier account models
        self.account_models = {}
        for sup_enc in np.unique(y_sup_encoded):
            mask = y_sup_encoded == sup_enc
            X_sub = X_train[mask]
            accounts_sub = accounts[mask]

            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(accounts_sub)

            if len(np.unique(y_encoded)) == 1:
                self.account_models[sup_enc] = {
                    "model": None, "encoder": encoder, "is_single": True, "fallback": accounts_sub[0]
                }
                continue

            X_res, y_res = SMOTE(random_state=self.random_state).fit_resample(X_sub, y_encoded)
            model = XGBClassifier(
                learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                eval_metric='mlogloss', random_state=self.random_state, n_estimators=300
            )
            model.fit(X_res, y_res)
            self.account_models[sup_enc] = {
                "model": model, "encoder": encoder, "is_single": False
            }

        s_f1, a_f1, joint = self._evaluate_hierarchical_fold(X_val, y_val)
        self.cv_metrics['supplier_f1'].append(s_f1)
        self.cv_metrics['account_f1'].append(a_f1)
        self.cv_metrics['joint_acc'].append(joint)
        print(f"Fold {fold+1} - Supplier F1: {s_f1:.4f} | Account F1: {a_f1:.4f} | Joint Accuracy: {joint:.4f}")


    """Evaluate (supplier, account) combined classification performance."""

    def _evaluate_combined_classification(self, y_true, y_pred):
        """Evaluates (supplier, account) combined as class labels."""
        combined_true = [f"{s}_{a}" for s, a in y_true]
        combined_pred = [f"{s}_{a}" for s, a in y_pred]

        print("\n📊 Classification Report for Combined (Supplier_Account) Labels:")
        print(classification_report(combined_true, combined_pred, zero_division=0))


    """Make predictions using the trained hierarchical model."""

    def predict(self, X, use_ensemble=True):
        if self.mode == 'hierarchical':
            return self._predict_hierarchical(X)

        raise ValueError("Invalid mode.")

    """Internal: Predict supplier and account using hierarchical logic."""

    def _predict_hierarchical(self, X):
        supplier_preds_encoded = self.supplier_model.predict(X)
        final_predictions = []
        for i, sup_enc in enumerate(supplier_preds_encoded):
            supplier_label = self.supplier_encoder.inverse_transform([sup_enc])[0]
            account_model_data = self.account_models.get(sup_enc)

            if not account_model_data:
                account_pred = "UNKNOWN"
            elif account_model_data["is_single"]:
                account_pred = account_model_data["fallback"]
            else:
                y_pred_encoded = account_model_data["model"].predict(X[i].reshape(1, -1))
                account_pred = account_model_data["encoder"].inverse_transform(y_pred_encoded)[0]

            final_predictions.append((supplier_label, account_pred))
        return final_predictions

    """Internal: Evaluate a validation fold for the hierarchical model."""

    def _evaluate_hierarchical_fold(self, X, y_true):
        predictions = self._predict_hierarchical(X)
        y_true_sup, y_true_acc = y_true[:, 0], y_true[:, 1]
        y_pred_sup = [p[0] for p in predictions]
        y_pred_acc = [p[1] for p in predictions]
        supplier_f1 = f1_score(y_true_sup, y_pred_sup, average='macro')
        account_f1 = f1_score(y_true_acc, y_pred_acc, average='macro')
        joint = np.mean([(ts == ps) and (ta == pa)
                         for ts, ta, ps, pa in zip(y_true_sup, y_true_acc, y_pred_sup, y_pred_acc)])
        return supplier_f1, account_f1, joint


    def _print_classification_report(self, y_true, y_pred):
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        label_names = None
        if self.class_names:
            try:
                label_names = [self.class_names[i] for i in unique_labels]
            except IndexError:
                logging.warning("Mismatch between label indices and class_names.")
        print(classification_report(y_true, y_pred, labels=unique_labels, target_names=label_names))


    def _plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        cm = confusion_matrix(y_true, y_pred)
        labels = np.unique(np.concatenate([y_true, y_pred]))
        disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(12, 10))
        disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
        plt.title(title)
        plt.tight_layout()
        plt.show()



    def _plot_combined_confusion_matrix(self, y_true, y_pred, title="Combined (Supplier_Account) Confusion Matrix"):
        combined_true = [f"{s}_{a}" for s, a in y_true]
        combined_pred = [f"{s}_{a}" for s, a in y_pred]

        labels = sorted(list(set(combined_true + combined_pred)))
        cm = confusion_matrix(combined_true, combined_pred, labels=labels)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(12, 10))
        disp.plot(ax=ax, cmap='Blues', xticks_rotation=90)
        plt.title(title)
        plt.tight_layout()
        plt.show()


    """Plot the top N most important features."""

    def _plot_feature_importance(self, top_n=15, save_path=None):
        if self.feature_importances_ is None or self.feature_names is None:
            return

        indices = np.argsort(self.feature_importances_)[::-1][:top_n]
        feature_labels = [self.feature_names[i] for i in indices]

        plt.figure(figsize=(12, 6))
        plt.title(f"Top {top_n} Feature Importance")
        plt.bar(range(top_n), self.feature_importances_[indices])
        plt.xticks(range(top_n), feature_labels, rotation=45, ha='right')
        plt.tight_layout()    def _store_feature_importances(self, model):
        importances = model.feature_importances_
        if self.feature_importances_ is None:
            self.feature_importances_ = importances
        else:
            self.feature_importances_ += importancesdef save_model(self, path):
        data = {
            'mode': self.mode,
            'cv_metrics': self.cv_metrics,
            'feature_importances': self.feature_importances_,
            'class_names': self.class_names,
            'feature_names': self.feature_names
        }
        if self.mode == 'hierarchical':
            data['supplier_model'] = self.supplier_model
            data['account_models'] = self.account_models
            data['supplier_encoder'] = self.supplier_encoder
        joblib.dump(data, path)

    @classmethod

    def load_model(cls, path):
        data = joblib.load(path)
        manager = cls(mode=data.get('mode', 'ensemble'))
        manager.cv_metrics = data['cv_metrics']
        manager.feature_importances_ = data['feature_importances']
        manager.class_names = data['class_names']
        manager.feature_names = data.get('feature_names')
        if manager.mode == 'hierarchical':
            manager.supplier_model = data['supplier_model']
            manager.account_models = data['account_models']
            manager.supplier_encoder = data['supplier_encoder']
        return manager