#!/usr/bin/env python
"""
COMPLETE REFACTOR PIPELINE
===========================
Trains the refactored XGBoost model using ONLY nutrition label features.

This script:
1. Loads and cleans the nutrition dataset
2. Extracts features from nutrition labels (no images, brands, etc.)
3. Trains XGBoost model on the cleaned data
4. Saves trained model artifacts
5. Reports accuracy and metrics

Usage:
    python COMPLETE_REFACTOR_PIPELINE.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import joblib
import re
import json
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("COMPLETE REFACTOR PIPELINE - Nutrition Label Features Only")
print("=" * 80)

# ============================================================================
# 1. LOAD AND INSPECT DATA
# ============================================================================

print("\n[1/6] Loading dataset...")
df = pd.read_csv('preprocessedPhase3FoodFacts.csv')
print(f"  ✓ Loaded {len(df)} products, {len(df.columns)} columns")

# ============================================================================
# 2. CLEAN DATA - REMOVE FORBIDDEN FEATURES
# ============================================================================

print("\n[2/6] Cleaning data (removing non-label features)...")

FORBIDDEN_FEATURES = {
    'image_path', 'image_160_path', 'has_image', 'has_image_160',
    'nova_group', 'ecoscore_grade', 'ecoscore_score', 'carbon_footprint_100g',
    'nutrient_level_fat', 'nutrient_level_saturated_fat',
    'nutrient_level_sugars', 'nutrient_level_salt',
    'healthy_score', 'macro_balance', 'protein_ratio', 'sugar_ratio',
    'log_energy_kcal_100g', 'log_fat_100g', 'log_sugars_100g', 'log_salt_100g',
    'brand', 'brand_cleaned', 'countries', 'countries_cleaned',
    'stores', 'origins', 'manufacturing_places', 'packaging',
    'additives', 'additives_cleaned', 'url', 'product_name', 'barcode',
    'quantity', 'serving_size', 'allergens', 'traces', 'main_image_url',
    'categories', 'vegetarian_status', 'vegan_status', 'allergens_cleaned',
    'product_id'
}

# Keep ONLY nutrition label features
REQUIRED_FEATURES = {
    'energy_kcal_100g', 'fat_100g', 'saturated_fat_100g',
    'carbohydrates_100g', 'sugars_100g', 'fiber_100g',
    'proteins_100g', 'salt_100g', 'ingredients_text_cleaned',
    'nutriscore_letter'
}

# Select required columns
available_features = [f for f in REQUIRED_FEATURES if f in df.columns]
df_clean = df[available_features].copy()

print(f"  ✓ Selected {len(available_features)} features from nutrition labels")

# Remove rows with missing target
df_clean = df_clean[df_clean['nutriscore_letter'].notna()]
print(f"  ✓ Removed rows with missing Nutri-Score: {len(df)} → {len(df_clean)}")

# Remove rows with missing critical nutrition values
critical_nutrients = ['energy_kcal_100g', 'fat_100g', 'carbohydrates_100g', 'sugars_100g', 'proteins_100g']
for nutrient in critical_nutrients:
    df_clean = df_clean[df_clean[nutrient].notna()]

print(f"  ✓ Removed rows with missing critical nutrients: {len(df_clean)} remaining")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

print("\n[3/6] Engineering features...")

# Target: Convert original Nutri-Score (1-5) to classes (0-4)
target_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
reverse_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

df_clean['nutriscore_numeric'] = df_clean['nutriscore_letter'].map(target_mapping)

# Remove any rows where mapping failed
df_clean = df_clean[df_clean['nutriscore_numeric'].notna()]

print(f"  ✓ Target variable prepared: {len(df_clean)} products")

# Numeric features
numeric_cols = ['energy_kcal_100g', 'fat_100g', 'saturated_fat_100g',
                'carbohydrates_100g', 'sugars_100g', 'fiber_100g',
                'proteins_100g', 'salt_100g']

# Binary ingredient features
def count_additives(text):
    if not text or pd.isna(text):
        return 0
    e_numbers = re.findall(r'E\d+', str(text).upper())
    return len(e_numbers)

def has_keyword(text, keywords):
    if not text or pd.isna(text):
        return 0
    text = str(text).lower()
    return 1 if any(kw in text for kw in keywords) else 0

# Extract binary features
df_clean['additives_count'] = df_clean['ingredients_text_cleaned'].apply(count_additives)
df_clean['has_sugar'] = df_clean['ingredients_text_cleaned'].apply(
    lambda x: has_keyword(x, ['sugar', 'sucre', 'azúcar', 'zucchero', 'xucar', 'glucosa', 'glucose'])
)
df_clean['has_syrup'] = df_clean['ingredients_text_cleaned'].apply(
    lambda x: has_keyword(x, ['syrup', 'sirop', 'jarabe', 'sciroppo', 'fructose'])
)
df_clean['has_palm_oil'] = df_clean['ingredients_text_cleaned'].apply(
    lambda x: has_keyword(x, ['palm', 'palmitate', 'palmate'])
)
df_clean['has_artificial_color'] = df_clean['ingredients_text_cleaned'].apply(
    lambda x: has_keyword(x, ['e110', 'e102', 'e129', 'e133', 'e151', 'artificial color'])
)

binary_cols = ['additives_count', 'has_sugar', 'has_syrup', 'has_palm_oil', 'has_artificial_color']

print(f"  ✓ Created {len(binary_cols)} binary ingredient features")

# TF-IDF for ingredients text
df_clean['ingredients_text_cleaned'] = df_clean['ingredients_text_cleaned'].fillna('')

print(f"  ✓ Vectorizing ingredient text...")

# ============================================================================
# 4. PREPARE TRAINING DATA
# ============================================================================

print("\n[4/6] Preparing training data...")

# Split features and target
X_numeric = df_clean[numeric_cols].values
X_binary = df_clean[binary_cols].values
X_text = df_clean['ingredients_text_cleaned']
y = df_clean['nutriscore_numeric'].values

# Scale numeric features
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=50, stop_words='english', min_df=2)
X_text_vectorized = tfidf.fit_transform(X_text).toarray()

# Combine all features
X_combined = np.hstack([X_numeric_scaled, X_binary, X_text_vectorized])

print(f"  ✓ Feature matrix shape: {X_combined.shape}")
print(f"    - Numeric (scaled): {X_numeric_scaled.shape[1]}")
print(f"    - Binary: {X_binary.shape[1]}")
print(f"    - TF-IDF: {X_text_vectorized.shape[1]}")

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  ✓ Train-test split: {len(X_train)} train, {len(X_test)} test")

# ============================================================================
# 5. TRAIN MODEL
# ============================================================================

print("\n[5/6] Training XGBoost model...")

# Calculate class weights for imbalance handling
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y_train)

# Train model
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=5,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
    verbosity=0
)

model.fit(X_train, y_train, sample_weight=sample_weights)

print(f"  ✓ Model trained successfully")

# ============================================================================
# 6. EVALUATE AND SAVE
# ============================================================================

print("\n[6/6] Evaluating and saving...")

# Predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n  Model Performance:")
print(f"  ✓ Accuracy: {accuracy:.1%}")

# Save artifacts
model_dir = 'food_guardian_app/model_artifacts'

joblib.dump(model, f'{model_dir}/best_model_xgboost.pkl')
joblib.dump(scaler, f'{model_dir}/feature_scaler.pkl')
joblib.dump(tfidf, f'{model_dir}/tfidf_vectorizer.pkl')

# Save metadata
metadata = {
    'model_accuracy': float(accuracy),
    'numeric_cols': numeric_cols,
    'binary_feature_cols': binary_cols,
    'tfidf_feature_names': tfidf.get_feature_names_out().tolist(),
    'target_mapping': target_mapping,
    'reverse_mapping': reverse_mapping,
    'total_features': X_combined.shape[1],
    'training_samples': len(X_train),
    'test_samples': len(X_test),
}

with open(f'{model_dir}/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  ✓ Saved model artifacts to {model_dir}/")

# Print detailed report
print("\n" + "=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)

class_names = ['A (Best)', 'B', 'C', 'D', 'E (Worst)']
print(classification_report(y_test, y_pred, target_names=class_names))

print("=" * 80)
print("✓ TRAINING COMPLETE!")
print("=" * 80)
print(f"\nThe model is ready to use. Run:")
print(f"  cd food_guardian_app")
print(f"  python app.py")
print(f"\nThen open: http://localhost:5000")
