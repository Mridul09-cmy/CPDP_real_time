import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from bayes_opt import BayesianOptimization

# ==================== STEP 1: Load Data ====================
source_file = r"C:\Users\HP\OneDrive\Documents\s2_dataset_preprocessed.csv"
target_file = r"C:\Users\HP\OneDrive\Documents\synthetic_dataset_preprocessed.csv"
# Load datasets
source_data = pd.read_csv(source_file)
target_data = pd.read_csv(target_file)

# Identify Target Column
target_col = 'Defective'  # Ensure this is correct
X_source = source_data.drop(columns=[target_col])
y_source = source_data[target_col]

X_target = target_data.drop(columns=[target_col])
y_target = target_data[target_col]

# ==================== STEP 2: Feature Selection with RFE ====================
xgb_temp = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
selector = RFE(xgb_temp, n_features_to_select=10)
selector.fit(X_source, y_source)

selected_features = X_source.columns[selector.support_]
X_source = X_source[selected_features]
X_target = X_target[selected_features]

print(f"Selected Features: {selected_features}")

# ==================== STEP 3: Standardization ====================
scaler = StandardScaler()
X_source_scaled = scaler.fit_transform(X_source)
X_target_scaled = scaler.transform(X_target)

# Split for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_target_scaled, y_target, test_size=0.3, random_state=42)

# ==================== STEP 4: Compute Class Weights ====================
class_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# ==================== STEP 5: Bayesian Optimization for XGBoost ====================
def xgb_evaluate(learning_rate, max_depth, n_estimators, colsample_bytree, subsample):
    model = XGBClassifier(
        learning_rate=learning_rate,
        max_depth=int(max_depth),
        n_estimators=int(n_estimators),
        colsample_bytree=colsample_bytree,
        subsample=subsample,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),  # Class-weighted XGBoost
        random_state=42
    )
    model.fit(X_train, y_train, sample_weight=class_weights)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred_prob)

# Define parameter search space
param_bounds = {
    'learning_rate': (0.01, 0.3),
    'max_depth': (3, 10),
    'n_estimators': (50, 500),
    'colsample_bytree': (0.5, 1),
    'subsample': (0.5, 1)
}

optimizer = BayesianOptimization(f=xgb_evaluate, pbounds=param_bounds, random_state=42)
optimizer.maximize(init_points=5, n_iter=20)

# Get best parameters
best_params = optimizer.max['params']
best_params['max_depth'] = int(best_params['max_depth'])
best_params['n_estimators'] = int(best_params['n_estimators'])

print("\nBest Parameters from Bayesian Optimization:")
print(best_params)

# ==================== STEP 6: Train Final XGBoost Model ====================
final_model = XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),  # Class weighting instead of SMOTE
    random_state=42
)

final_model.fit(X_train, y_train, sample_weight=class_weights)
y_pred_prob = final_model.predict_proba(X_test)[:, 1]

# ==================== STEP 7: Evaluate Model Performance ====================
roc_auc = roc_auc_score(y_test, y_pred_prob)

print("\nFinal Model Performance:")
print(f"ROC-AUC Score: {roc_auc:.4f}")

y_pred = (y_pred_prob > 0.5).astype(int)
print("Classification Report:\n", classification_report(y_test, y_pred))