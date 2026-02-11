"""
Parsing Module
===============
Parses and structures OCR-extracted nutrition data.
Handles missing values, unit conversions, and data validation.
"""

import re
from typing import Dict, Any, Optional, List, Tuple
import numpy as np


class NutritionParser:
    """
    Parses and structures nutrition information extracted from labels.
    
    Features:
    - Unit conversion (mg to g, kJ to kcal, etc.)
    - Missing value imputation
    - Data validation
    - Handles user edits
    """
    
    # Common default values and ranges for validation
    DEFAULT_NOVA_GROUP = 2
    DEFAULT_ECOSCORE_GRADE = 'C'
    DEFAULT_ECOSCORE_SCORE = 50
    
    NUTRIENT_RANGES = {
        'fat_100g': (0, 100),
        'saturated_fat_100g': (0, 100),
        'carbohydrates_100g': (0, 100),
        'sugars_100g': (0, 100),
        'fiber_100g': (0, 100),
        'proteins_100g': (0, 100),
        'salt_100g': (0, 25),
    }
    
    NUTRIENT_LEVELS = {
        'fat': [10, 20, 35],  # Low, Moderate, High
        'saturated_fat': [1.5, 5, 10],
        'sugars': [5, 12.5, 25],
        'salt': [0.3, 0.9, 1.5],
    }
    
    def __init__(self):
        """Initialize the nutrition parser."""
        self.extraction_text = ""
        self.original_extraction = {}
        self.parsed_nutrition = {}
        self.user_edits = {}
    
    @staticmethod
    def mg_to_g(value_mg: float) -> float:
        """Convert milligrams to grams."""
        return value_mg / 1000.0
    
    @staticmethod
    def kj_to_kcal(value_kj: float) -> float:
        """Convert kilojoules to kilocalories."""
        return value_kj / 4.184
    
    @staticmethod
    def kcal_to_kj(value_kcal: float) -> float:
        """Convert kilocalories to kilojoules."""
        return value_kcal * 4.184
    
    def standardize_nutrition_values(self, 
                                     nutrition_values: Dict[str, Optional[float]]) -> Dict[str, float]:
        """
        Standardize nutrition values to per 100g basis and common units.
        
        Args:
            nutrition_values: Dictionary of raw nutrition values
        
        Returns:
            Standardized values
        """
        standardized = {}
        
        # Energy (convert kJ to kcal if needed)
        if nutrition_values.get('energy_kcal') is not None:
            standardized['energy_kcal_100g'] = float(nutrition_values['energy_kcal'])
        elif nutrition_values.get('energy_kj') is not None:
            standardized['energy_kcal_100g'] = self.kj_to_kcal(nutrition_values['energy_kj'])
        else:
            standardized['energy_kcal_100g'] = 0.0
        
        # Fat (already in grams)
        standardized['fat_100g'] = float(nutrition_values.get('fat_g') or 0)
        
        # Saturated fat
        standardized['saturated_fat_100g'] = float(nutrition_values.get('saturated_fat_g') or 0)
        
        # Carbohydrates
        standardized['carbohydrates_100g'] = float(nutrition_values.get('carbohydrates_g') or 0)
        
        # Sugars
        standardized['sugars_100g'] = float(nutrition_values.get('sugars_g') or 0)
        
        # Fiber
        standardized['fiber_100g'] = float(nutrition_values.get('fiber_g') or 0)
        
        # Protein
        standardized['proteins_100g'] = float(nutrition_values.get('protein_g') or 0)
        
        # Salt (convert sodium mg to salt g if needed)
        if nutrition_values.get('salt_g') is not None:
            standardized['salt_100g'] = float(nutrition_values['salt_g'])
        elif nutrition_values.get('sodium_mg') is not None:
            # Sodium to salt: salt = sodium * 2.54
            sodium_g = self.mg_to_g(nutrition_values['sodium_mg'])
            standardized['salt_100g'] = sodium_g * 2.54
        else:
            standardized['salt_100g'] = 0.0
        
        return standardized
    
    def _scale_to_per_100g(self, 
                          nutrition_values: Dict[str, Optional[float]], 
                          scaling_factor: float) -> Dict[str, Optional[float]]:
        """
        Scale nutrition values from per-serving to per-100g.
        
        Args:
            nutrition_values: Raw nutrition values (per serving)
            scaling_factor: Factor to multiply by (100 / serving_size_g)
        
        Returns:
            Scaled nutrition values (per 100g)
        """
        nutrient_fields = [
            'energy_kcal', 'energy_kj', 'fat_g', 'saturated_fat_g', 'trans_fat_g',
            'carbohydrates_g', 'sugars_g', 'fiber_g', 'protein_g', 'sodium_mg', 'salt_g'
        ]
        
        scaled = nutrition_values.copy()
        for field in nutrient_fields:
            if field in scaled and scaled[field] is not None:
                scaled[field] = float(scaled[field]) * scaling_factor
        
        return scaled
    
    def validate_nutrition_values(self, values: Dict[str, float]) -> Dict[str, Tuple[bool, str]]:
        """
        Validate nutrition values for reasonableness.
        
        Args:
            values: Standardized nutrition values
        
        Returns:
            Dictionary of validation results {nutrient: (is_valid, message)}
        """
        results = {}
        
        for nutrient, (min_val, max_val) in self.NUTRIENT_RANGES.items():
            value = values.get(nutrient, 0)
            is_valid = min_val <= value <= max_val
            
            message = f"✓ {nutrient}: {value:.2f}g"
            if not is_valid:
                message = f"⚠ {nutrient}: {value:.2f}g (outside range {min_val}-{max_val})"
            
            results[nutrient] = (is_valid, message)
        
        return results
    
    def compute_nutrient_levels(self, values: Dict[str, float]) -> Dict[str, str]:
        """
        Compute nutrient level categories (Low, Moderate, High).
        
        Args:
            values: Standardized nutrition values
        
        Returns:
            Mapping of nutrient -> level
        """
        levels = {}
        
        # Fat
        fat = values.get('fat_100g', 0)
        if fat <= self.NUTRIENT_LEVELS['fat'][0]:
            levels['fat'] = 'Low'
        elif fat <= self.NUTRIENT_LEVELS['fat'][1]:
            levels['fat'] = 'Moderate'
        else:
            levels['fat'] = 'High'
        
        # Saturated Fat
        sat_fat = values.get('saturated_fat_100g', 0)
        if sat_fat <= self.NUTRIENT_LEVELS['saturated_fat'][0]:
            levels['saturated_fat'] = 'Low'
        elif sat_fat <= self.NUTRIENT_LEVELS['saturated_fat'][1]:
            levels['saturated_fat'] = 'Moderate'
        else:
            levels['saturated_fat'] = 'High'
        
        # Sugars
        sugars = values.get('sugars_100g', 0)
        if sugars <= self.NUTRIENT_LEVELS['sugars'][0]:
            levels['sugars'] = 'Low'
        elif sugars <= self.NUTRIENT_LEVELS['sugars'][1]:
            levels['sugars'] = 'Moderate'
        else:
            levels['sugars'] = 'High'
        
        # Salt
        salt = values.get('salt_100g', 0)
        if salt <= self.NUTRIENT_LEVELS['salt'][0]:
            levels['salt'] = 'Low'
        elif salt <= self.NUTRIENT_LEVELS['salt'][1]:
            levels['salt'] = 'Moderate'
        else:
            levels['salt'] = 'High'
        
        return levels
    
    def compute_derived_features(self, values: Dict[str, float]) -> Dict[str, float]:
        """
        Compute derived nutritional features for model input.
        
        Args:
            values: Standardized nutrition values
        
        Returns:
            Dictionary of derived features
        """
        derived = {}
        
        # Nova Group (default to processed)
        derived['nova_group'] = self.DEFAULT_NOVA_GROUP
        
        # Health indicators
        total_macro = (values.get('fat_100g', 0) + 
                      values.get('carbohydrates_100g', 0) + 
                      values.get('proteins_100g', 0))
        
        if total_macro > 0:
            derived['protein_ratio'] = values.get('proteins_100g', 0) / total_macro
            derived['macro_balance'] = min(0.4, derived['protein_ratio'])  # Capped at 0.4
        else:
            derived['protein_ratio'] = 0.0
            derived['macro_balance'] = 0.0
        
        # Energy density
        energy = values.get('energy_kcal_100g', 0)
        derived['energy_density'] = energy / 900.0  # Normalized
        
        # Sugar ratio
        carbs = values.get('carbohydrates_100g', 0)
        if carbs > 0:
            derived['sugar_ratio'] = values.get('sugars_100g', 0) / carbs
        else:
            derived['sugar_ratio'] = 0.0
        
        # Log transformations
        derived['log_energy_kcal_100g'] = np.log1p(energy)
        derived['log_salt_100g'] = np.log1p(values.get('salt_100g', 0))
        
        # Healthy score (higher is healthier)
        healthy_score = 100.0
        healthy_score -= values.get('sugars_100g', 0) * 2  # Penalize sugar
        healthy_score -= values.get('saturated_fat_100g', 0) * 1.5  # Penalize sat fat
        healthy_score -= values.get('salt_100g', 0) * 10  # Penalize salt
        healthy_score += values.get('fiber_100g', 0) * 3  # Reward fiber
        healthy_score += values.get('proteins_100g', 0) * 1.5  # Reward protein
        
        derived['healthy_score'] = max(0, min(100, healthy_score))
        
        # Dummy values for other required fields
        derived['contains_palm_oil'] = 0  # Assume no by default
        derived['vegetarian_status'] = 1  # Assume yes by default
        derived['vegan_status'] = 0  # Assume no by default
        derived['nutrient_level_fat'] = 1  # Map to moderate
        derived['nutrient_level_saturated_fat'] = 1
        derived['nutrient_level_sugars'] = 1
        derived['nutrient_level_salt'] = 1
        derived['ecoscore_grade'] = 2  # Neutral (C)
        derived['ecoscore_score'] = self.DEFAULT_ECOSCORE_SCORE
        derived['carbon_footprint_100g'] = 1000  # Default estimate
        derived['additives_count'] = 0  # Default
        
        return derived
    
    def parse_ocr_extraction(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full parsing pipeline from OCR result to structured nutrition data.
        
        Handles serving size scaling: if serving_size is provided and not 100g,
        scales all nutrition values from per-serving to per-100g.
        
        Args:
            ocr_result: Output from OCR module
        
        Returns:
            Comprehensive parsed result
        """
        self.original_extraction = ocr_result.copy()
        self.extraction_text = ocr_result.get('cleaned_text', '')
        
        # Extract raw nutrition and serving size
        raw_nutrition = ocr_result.get('nutrition_values', {})
        serving_size_g = ocr_result.get('serving_size', None)
        
        # If serving size is provided and not 100g, scale nutrition values
        if serving_size_g and serving_size_g != 100:
            scaling_factor = 100.0 / serving_size_g
            raw_nutrition = self._scale_to_per_100g(raw_nutrition, scaling_factor)
        
        # Standardize nutrition values
        standardized = self.standardize_nutrition_values(raw_nutrition)
        
        # Validate
        validation = self.validate_nutrition_values(standardized)
        
        # Compute nutrient levels
        nutrient_levels = self.compute_nutrient_levels(standardized)
        
        # Compute derived features
        derived = self.compute_derived_features(standardized)
        
        # Combine all nutrition data
        self.parsed_nutrition = {**standardized, **derived}
        
        return {
            'success': True,
            'standardized_nutrition': standardized,
            'validation': validation,
            'nutrient_levels': nutrient_levels,
            'derived_features': derived,
            'full_nutrition_dict': self.parsed_nutrition,
            'ingredients': ocr_result.get('ingredients', ''),
            'allergens': ocr_result.get('allergens', ''),
            'raw_text': ocr_result.get('raw_text', ''),
            'serving_size': serving_size_g
        }
    
    def apply_user_edits(self, edits: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply user-provided edits to parsed nutrition data.
        
        Args:
            edits: Dictionary of {field: new_value} edits
        
        Returns:
            Updated parsed nutrition result
        """
        self.user_edits = edits.copy()
        
        # Update raw nutrition values if provided
        for key, value in edits.items():
            if key in self.parsed_nutrition:
                try:
                    self.parsed_nutrition[key] = float(value)
                except (ValueError, TypeError):
                    print(f"Invalid value for {key}: {value}")
        
        # Recompute derived features after edits
        standardized = {
            k: v for k, v in self.parsed_nutrition.items()
            if any(nutrient in k for nutrient in 
                   ['fat', 'carbohydrates', 'sugars', 'fiber', 'proteins', 'salt', 'energy'])
        }
        
        if standardized:
            derived = self.compute_derived_features(standardized)
            self.parsed_nutrition.update(derived)
        
        return self.parsed_nutrition
    
    def get_input_for_model(self) -> Dict[str, float]:
        """
        Get properly formatted input dictionary for model prediction.
        
        Returns:
            Dictionary ready for model.predict()
        """
        # Text inputs (leave empty - would be filled from OCR)
        text_inputs = {
            'brand': '',
            'allergens': self.original_extraction.get('allergens', ''),
            'ingredients_text': self.original_extraction.get('ingredients', ''),
            'countries': '',
            'additives': ''
        }
        
        # Tabular inputs
        tabular_inputs = {
            'nova_group': self.parsed_nutrition.get('nova_group', 2),
            'fat_100g': self.parsed_nutrition.get('fat_100g', 0),
            'saturated_fat_100g': self.parsed_nutrition.get('saturated_fat_100g', 0),
            'carbohydrates_100g': self.parsed_nutrition.get('carbohydrates_100g', 0),
            'sugars_100g': self.parsed_nutrition.get('sugars_100g', 0),
            'fiber_100g': self.parsed_nutrition.get('fiber_100g', 0),
            'proteins_100g': self.parsed_nutrition.get('proteins_100g', 0),
            'contains_palm_oil': self.parsed_nutrition.get('contains_palm_oil', 0),
            'vegetarian_status': self.parsed_nutrition.get('vegetarian_status', 1),
            'vegan_status': self.parsed_nutrition.get('vegan_status', 0),
            'nutrient_level_fat': self.parsed_nutrition.get('nutrient_level_fat', 1),
            'nutrient_level_saturated_fat': self.parsed_nutrition.get('nutrient_level_saturated_fat', 1),
            'nutrient_level_sugars': self.parsed_nutrition.get('nutrient_level_sugars', 1),
            'nutrient_level_salt': self.parsed_nutrition.get('nutrient_level_salt', 1),
            'ecoscore_grade': self.parsed_nutrition.get('ecoscore_grade', 2),
            'ecoscore_score': self.parsed_nutrition.get('ecoscore_score', 50),
            'carbon_footprint_100g': self.parsed_nutrition.get('carbon_footprint_100g', 1000),
            'additives_count': self.parsed_nutrition.get('additives_count', 0),
            'sugar_ratio': self.parsed_nutrition.get('sugar_ratio', 0),
            'energy_density': self.parsed_nutrition.get('energy_density', 0),
            'protein_ratio': self.parsed_nutrition.get('protein_ratio', 0),
            'macro_balance': self.parsed_nutrition.get('macro_balance', 0),
            'healthy_score': self.parsed_nutrition.get('healthy_score', 50),
            'log_energy_kcal_100g': self.parsed_nutrition.get('log_energy_kcal_100g', 0),
            'log_salt_100g': self.parsed_nutrition.get('log_salt_100g', 0)
        }
        
        return {
            'text_inputs': text_inputs,
            'tabular_inputs': tabular_inputs
        }


# Global parser instance
_parser = None


def get_parser() -> NutritionParser:
    """
    Get or create parser instance (singleton).
    
    Returns:
        NutritionParser instance
    """
    global _parser
    if _parser is None:
        _parser = NutritionParser()
    return _parser
