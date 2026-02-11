"""
Model Loader Module
===================
Loads and manages the trained XGBoost model along with preprocessors.
Handles model inference and feature extraction from text and tabular data.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# TensorFlow/Keras for embeddings
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


class ModelLoader:
    """
    Loads and manages the trained XGBoost model with text and tabular components.
    
    This class handles:
    - Loading the trained XGBoost model
    - Loading preprocessed text and tabular encoders (tokenizer, scaler)
    - Loading neural network feature extractors for text and tabular data
    - Running inference on new data
    """
    
    def __init__(self, model_dir: str = "./model_artifacts"):
        """
        Initialize the model loader.
        
        Args:
            model_dir: Directory containing model artifacts
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Model components
        self.xgb_model = None
        self.text_model = None
        self.tokenizer = None
        self.scaler = None
        self.text_extractor = None
        self.tab_extractor = None
        
        # Configuration
        self.max_len = 220
        self.vocab_size = 30000
        self.num_classes = 5
        self.nutri_score_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
        self.nutri_score_reverse = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        
        # Text and tabular columns configuration
        self.text_cols = [
            "brand",
            "allergens",
            "ingredients_text",
            "countries",
            "additives",
        ]
        
        self.tabular_cols = [
            'nova_group', 'fat_100g', 'saturated_fat_100g', 'carbohydrates_100g',
            'sugars_100g', 'fiber_100g', 'proteins_100g', 'contains_palm_oil',
            'vegetarian_status', 'vegan_status', 'nutrient_level_fat',
            'nutrient_level_saturated_fat', 'nutrient_level_sugars',
            'nutrient_level_salt', 'ecoscore_grade', 'ecoscore_score',
            'carbon_footprint_100g', 'additives_count', 'sugar_ratio',
            'energy_density', 'protein_ratio', 'macro_balance', 'healthy_score',
            'log_energy_kcal_100g', 'log_salt_100g'
        ]
    
    def save_model_artifacts(self, 
                            xgb_model,
                            tokenizer,
                            scaler,
                            text_extractor,
                            tab_extractor):
        """
        Save model artifacts to disk.
        
        Args:
            xgb_model: Trained XGBoost model
            tokenizer: Keras tokenizer for text processing
            scaler: StandardScaler for tabular data
            text_extractor: Text embedding extractor model
            tab_extractor: Tabular embedding extractor model
        """
        # Save XGBoost model
        with open(self.model_dir / "xgb_model.pkl", "wb") as f:
            pickle.dump(xgb_model, f)
        
        # Save tokenizer
        with open(self.model_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        
        # Save scaler
        with open(self.model_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        
        # Save neural network models
        text_extractor.save(self.model_dir / "text_extractor.h5")
        tab_extractor.save(self.model_dir / "tab_extractor.h5")
        
        print(f"✓ Model artifacts saved to {self.model_dir}")
    
    def load_model_artifacts(self) -> bool:
        """
        Load model artifacts from disk.
        
        Returns:
            True if all artifacts loaded successfully, False otherwise
        """
        try:
            # Load XGBoost model
            with open(self.model_dir / "xgb_model.pkl", "rb") as f:
                self.xgb_model = pickle.load(f)
            
            # Load tokenizer
            with open(self.model_dir / "tokenizer.pkl", "rb") as f:
                self.tokenizer = pickle.load(f)
            
            # Load scaler
            with open(self.model_dir / "scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            
            # Load neural network models
            self.text_extractor = tf.keras.models.load_model(
                self.model_dir / "text_extractor.h5"
            )
            self.tab_extractor = tf.keras.models.load_model(
                self.model_dir / "tab_extractor.h5"
            )
            
            print("✓ All model artifacts loaded successfully")
            return True
            
        except FileNotFoundError as e:
            print(f"✗ Error loading model artifacts: {e}")
            return False
    
    def preprocess_text(self, text_input: Dict[str, str]) -> np.ndarray:
        """
        Preprocess text input using tokenizer and padding.
        
        Args:
            text_input: Dictionary with text fields (brand, allergens, etc.)
        
        Returns:
            Padded token sequences
        """
        # Concatenate text fields
        concatenated = ""
        for col in self.text_cols:
            value = str(text_input.get(col, "")).strip()
            if value and value != "nan":
                concatenated += " " + value
        
        # Tokenize and pad
        sequences = self.tokenizer.texts_to_sequences([concatenated])
        padded = pad_sequences(sequences, maxlen=self.max_len, 
                              padding="post", truncating="post")
        
        return padded
    
    def preprocess_tabular(self, tabular_input: Dict[str, float]) -> np.ndarray:
        """
        Preprocess tabular input using scaler.
        
        Args:
            tabular_input: Dictionary with numeric features
        
        Returns:
            Scaled tabular features
        """
        # Create feature vector in correct order
        features = []
        for col in self.tabular_cols:
            value = tabular_input.get(col, 0.0)
            # Handle missing values
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = 0.0
            features.append(float(value))
        
        # Scale features
        features_array = np.array([features]).astype(np.float32)
        scaled = self.scaler.transform(features_array).astype(np.float32)
        
        return scaled
    
    def extract_embeddings(self, 
                          text_padded: np.ndarray,
                          tabular_scaled: np.ndarray) -> np.ndarray:
        """
        Extract embeddings from text and tabular data.
        
        Args:
            text_padded: Padded text sequences
            tabular_scaled: Scaled tabular features
        
        Returns:
            Fused embeddings (text + tabular concatenated)
        """
        # Extract text embeddings
        text_emb = self.text_extractor.predict(text_padded, verbose=0)
        
        # Extract tabular embeddings
        tab_emb = self.tab_extractor.predict(tabular_scaled, verbose=0)
        
        # Fuse embeddings (concatenate)
        fused = np.hstack([text_emb, tab_emb]).astype(np.float32)
        
        # Clean any NaN/infinite values
        fused = np.nan_to_num(fused, nan=0.0, 
                             posinf=np.finfo(np.float32).max,
                             neginf=np.finfo(np.float32).min)
        
        return fused
    
    def predict(self, text_input: Dict[str, str], 
                tabular_input: Dict[str, float]) -> Dict[str, Any]:
        """
        Make prediction on new data.
        
        Args:
            text_input: Dictionary with text features
            tabular_input: Dictionary with numeric features
        
        Returns:
            Dictionary with prediction and confidence scores
        """
        if self.xgb_model is None:
            raise RuntimeError("Model not loaded. Call load_model_artifacts() first.")
        
        # Preprocess inputs
        text_padded = self.preprocess_text(text_input)
        tabular_scaled = self.preprocess_tabular(tabular_input)
        
        # Extract embeddings
        fused = self.extract_embeddings(text_padded, tabular_scaled)
        
        # Make prediction
        prediction = self.xgb_model.predict(fused)[0]
        probabilities = self.xgb_model.predict_proba(fused)[0]
        
        # Convert to Nutri-Score letter
        nutri_score_letter = self.nutri_score_map[int(prediction)]
        
        # Get confidence
        confidence = float(np.max(probabilities))
        
        # Get per-class probabilities
        class_scores = {
            self.nutri_score_map[i]: float(prob)
            for i, prob in enumerate(probabilities)
        }
        
        return {
            'prediction': prediction,
            'nutri_score': nutri_score_letter,
            'confidence': confidence,
            'class_probabilities': class_scores,
            'all_scores': probabilities.tolist()
        }
    
    def batch_predict(self, 
                     text_inputs: list,
                     tabular_inputs: list) -> list:
        """
        Make batch predictions.
        
        Args:
            text_inputs: List of text input dictionaries
            tabular_inputs: List of tabular input dictionaries
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for text_input, tab_input in zip(text_inputs, tabular_inputs):
            result = self.predict(text_input, tab_input)
            results.append(result)
        
        return results
    
    def get_feature_names(self) -> Tuple[list, list]:
        """
        Get feature names for model.
        
        Returns:
            Tuple of (text_cols, tabular_cols)
        """
        return self.text_cols, self.tabular_cols


# Singleton instance for global access
_model_loader = None


def get_model_loader(model_dir: str = "./model_artifacts") -> ModelLoader:
    """
    Get or create a model loader instance (singleton pattern).
    
    Args:
        model_dir: Directory containing model artifacts
    
    Returns:
        ModelLoader instance
    """
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader(model_dir)
    return _model_loader
