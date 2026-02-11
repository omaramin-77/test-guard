"""
Refactored Model Loader - Uses ONLY nutrition label features
=============================================================
Loads best_model_xgboost.pkl and preprocessing pipeline.
Rejects any old/forbidden features to prevent legacy behavior.

Features accepted:
- energy_kcal_100g, fat_100g, saturated_fat_100g, carbohydrates_100g
- sugars_100g, fiber_100g, proteins_100g, salt_100g
- ingredients_text_cleaned

DO NOT use: brand, country, ecoscore, image, nova_group, etc.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import re
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
from sklearn.preprocessing import StandardScaler

class RefactoredModelLoader:
    """
    Loads and manages the refactored XGBoost model.
    Uses ONLY nutrition label features - rejects legacy features.
    """
    
    # Forbidden features that MUST NOT be used
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
    
    def __init__(self, model_dir: str = "./model_artifacts"):
        """Initialize the model loader."""
        self.model_dir = Path(model_dir)
        
        # Model components
        self.xgb_model = None
        self.scaler = None
        self.tfidf_vectorizer = None
        self.metadata = None
        
        # Load all components
        self.load_all_artifacts()
    
    def load_all_artifacts(self):
        """Load model and preprocessing artifacts."""
        try:
            # Load model
            model_path = self.model_dir / 'best_model_xgboost.pkl'
            self.xgb_model = joblib.load(model_path)
            print(f"✓ Loaded XGBoost model from {model_path}")
            
            # Load scaler
            scaler_path = self.model_dir / 'feature_scaler.pkl'
            self.scaler = joblib.load(scaler_path)
            print(f"✓ Loaded feature scaler from {scaler_path}")
            
            # Load TF-IDF vectorizer
            tfidf_path = self.model_dir / 'tfidf_vectorizer.pkl'
            self.tfidf_vectorizer = joblib.load(tfidf_path)
            print(f"✓ Loaded TF-IDF vectorizer from {tfidf_path}")
            
            # Load metadata
            metadata_path = self.model_dir / 'model_metadata.json'
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"✓ Loaded metadata from {metadata_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model artifacts: {e}")
    
    def validate_input_features(self, input_data: Dict[str, Any]):
        """
        Validate that input contains ONLY allowed features.
        Fail loudly if forbidden features are passed.
        
        Args:
            input_data: Dictionary of input features
            
        Raises:
            ValueError: If forbidden features are detected
        """
        input_keys = set(input_data.keys())
        forbidden_found = input_keys & self.FORBIDDEN_FEATURES
        
        if forbidden_found:
            raise ValueError(
                f"❌ FORBIDDEN FEATURES DETECTED: {forbidden_found}\n"
                f"This model uses ONLY nutrition label features.\n"
                f"Remove these legacy features before prediction."
            )
    
    def preprocess_nutrition_values(self, nutrition_data: Dict[str, float]) -> Dict[str, float]:
        """
        Preprocess nutrition values with realistic ranges.
        
        Args:
            nutrition_data: Raw nutrition values from label
            
        Returns:
            Clipped nutrition values
        """
        ranges = {
            'energy_kcal_100g': (0, 900),
            'fat_100g': (0, 100),
            'saturated_fat_100g': (0, 100),
            'carbohydrates_100g': (0, 100),
            'sugars_100g': (0, 100),
            'fiber_100g': (0, 50),
            'proteins_100g': (0, 100),
            'salt_100g': (0, 20),
        }
        
        processed = nutrition_data.copy()
        for key, (min_val, max_val) in ranges.items():
            if key in processed:
                value = processed[key]
                if pd.notna(value):
                    processed[key] = np.clip(float(value), min_val, max_val)
                else:
                    if key == 'fiber_100g':
                        processed[key] = 0.0
        
        return processed
    
    def count_additives(self, ingredients_text: str) -> int:
        """Count E-numbers in ingredient text."""
        if not ingredients_text:
            return 0
        e_numbers = re.findall(r'E\d+', str(ingredients_text).upper())
        return len(e_numbers)
    
    def has_ingredient(self, ingredients_text: str, keywords: list) -> int:
        """Check if ingredient appears in text."""
        if not ingredients_text:
            return 0
        text = str(ingredients_text).lower()
        return 1 if any(kw in text for kw in keywords) else 0
    
    def extract_ingredient_features(self, ingredients_text: str) -> Dict[str, int]:
        """Extract binary ingredient flags."""
        if not ingredients_text:
            ingredients_text = ''
        
        sugar_kw = ['sugar', 'sucre', 'azúcar', 'zucchero', 'xucar', 'glucosa', 'glucose', 'dextrose']
        syrup_kw = ['syrup', 'sirop', 'jarabe', 'sciroppo', 'fructose', 'glucose syrup', 'corn syrup']
        palm_kw = ['palm', 'palmitate', 'palmate', 'huile de palme', 'olio di palma', 'aceite de palma']
        color_kw = ['e110', 'e102', 'e129', 'e133', 'e151', 'tartrazine', 'allura', 'sunset yellow',
                   'carmine', 'cochineal', 'colorant', 'artificial color', 'synthetic color']
        
        return {
            'additives_count': self.count_additives(ingredients_text),
            'has_sugar': self.has_ingredient(ingredients_text, sugar_kw),
            'has_syrup': self.has_ingredient(ingredients_text, syrup_kw),
            'has_palm_oil': self.has_ingredient(ingredients_text, palm_kw),
            'has_artificial_color': self.has_ingredient(ingredients_text, color_kw),
        }
    
    def predict(self, nutrition_data: Dict[str, float], ingredients_text: str) -> Dict[str, Any]:
        """
        Make prediction for a product.
        
        Args:
            nutrition_data: Nutrition values (per 100g)
            ingredients_text: Ingredient list text
            
        Returns:
            Prediction with Nutri-Score letter and confidence
        """
        # Validate no forbidden features
        self.validate_input_features(nutrition_data)
        
        # Preprocess nutrition values
        nutrition_data = self.preprocess_nutrition_values(nutrition_data)
        
        # Extract ingredient features
        ingredient_features = self.extract_ingredient_features(ingredients_text)
        
        # Prepare feature vector
        numeric_cols = self.metadata['numeric_cols']
        binary_cols = self.metadata['binary_feature_cols']
        tfidf_names = self.metadata['tfidf_feature_names']
        
        # Build numeric features
        numeric_values = []
        for col in numeric_cols:
            value = nutrition_data.get(col, 0)
            if pd.isna(value):
                value = 0 if col == 'fiber_100g' else np.nan
            numeric_values.append(value)
        
        numeric_array = np.array(numeric_values).reshape(1, -1)
        
        # Scale numeric features
        numeric_scaled = self.scaler.transform(numeric_array)[0]
        
        # Build binary features
        binary_values = [ingredient_features[col] for col in binary_cols]
        
        # Get TF-IDF features
        tfidf_sparse = self.tfidf_vectorizer.transform([ingredients_text])
        tfidf_values = tfidf_sparse.toarray()[0]
        
        # Combine all features
        all_features = np.concatenate([numeric_scaled, binary_values, tfidf_values]).reshape(1, -1)
        
        # Make prediction
        pred_numeric = self.xgb_model.predict(all_features)[0]
        pred_proba = self.xgb_model.predict_proba(all_features)[0]
        
        # Convert to letter
        reverse_map = self.metadata['reverse_mapping']
        pred_letter = reverse_map[str(int(pred_numeric))]
        
        # Get confidence (max probability)
        confidence = float(np.max(pred_proba))
        
        return {
            'nutri_score': pred_letter,
            'confidence': confidence,
            'predictions': {
                reverse_map[str(i)]: float(prob)
                for i, prob in enumerate(pred_proba)
            }
        }


# Singleton instance
_model_loader = None

def get_model_loader(model_dir: str = "./model_artifacts") -> RefactoredModelLoader:
    """Get or create model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = RefactoredModelLoader(model_dir)
    return _model_loader
