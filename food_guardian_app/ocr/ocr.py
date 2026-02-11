"""
OCR Module - OCR.space Implementation
======================================
Main OCR interface using OCR.space free API.

This module:
- Provides backward-compatible interface
- Uses OCR.space for text extraction
- Handles parsing robustly
- Never breaks the application

The OCR quality is delegated to OCR.space.
This module focuses on output normalization and parsing.
"""

import logging
from typing import Dict, Optional
import re
from ocr.ocr_space_client import get_ocr_client

logger = logging.getLogger(__name__)


class NutritionOCR:
    """
    OCR interface for nutrition labels (using OCR.space API).
    
    Maintains compatibility with existing code while using
    the robust OCR.space service for text extraction.
    """
    
    def __init__(self):
        """Initialize OCR engine."""
        self.ocr_client = get_ocr_client()
        logger.info("NutritionOCR initialized with OCR.space API")
    
    def full_extraction(self, image_path: str) -> Dict[str, any]:
        """
        Full extraction pipeline from image to structured data.
        
        Pipeline:
        1. Call OCR.space API
        2. Normalize and clean text
        3. Extract nutrition values
        4. Extract ingredients
        5. Extract allergens
        
        Args:
            image_path: Path to nutrition label image
        
        Returns:
            Comprehensive extraction result
        """
        logger.info(f"{'='*70}")
        logger.info(f"Starting OCR extraction: {image_path}")
        logger.info(f"{'='*70}")
        
        # Step 1: Call OCR.space API
        ocr_result = self.ocr_client.extract_text_from_image(image_path)
        
        if not ocr_result['success']:
            logger.error(f"OCR failed: {ocr_result['error_message']}")
            return {
                'success': False,
                'error': ocr_result['error_message'],
                'raw_text': '',
                'cleaned_text': '',
                'nutrition_values': {},
                'ingredients': '',
                'allergens': ''
            }
        
        raw_text = ocr_result['raw_text']
        logger.info(f"OCR successful: {len(raw_text)} chars, confidence: {ocr_result['confidence']:.0f}%")
        
        # Step 2: Normalize and clean text
        cleaned_text = self._clean_text(raw_text)
        logger.info(f"Cleaned text: {len(cleaned_text)} chars")
        
        # Step 3: Extract serving size
        serving_size = self._extract_serving_size(cleaned_text)
        if serving_size:
            logger.info(f"Serving size: {serving_size}g")
        
        # Step 4: Extract nutrition values
        nutrition_values = self._extract_nutrition_values(cleaned_text)
        extracted_nutrients = sum(1 for v in nutrition_values.values() if v is not None)
        logger.info(f"Extracted {extracted_nutrients} nutrition values")
        
        # Step 5: Extract ingredients
        ingredients = self._extract_ingredients(cleaned_text)
        logger.info(f"Ingredients: {'Found' if ingredients else 'Not found'}")
        
        # Step 6: Extract allergens
        allergens = self._extract_allergens(cleaned_text)
        logger.info(f"Allergens: {'Found' if allergens else 'Not found'}")
        
        logger.info(f"{'='*70}")
        
        return {
            'success': True,
            'raw_text': raw_text,
            'cleaned_text': cleaned_text,
            'nutrition_values': nutrition_values,
            'serving_size': serving_size,
            'ingredients': ingredients,
            'allergens': allergens,
            'ocr_confidence': ocr_result['confidence']
        }
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean OCR output text.
        
        Handles:
        - Whitespace normalization
        - Common OCR artifacts
        - Special character cleanup
        
        Args:
            text: Raw OCR text
        
        Returns:
            Cleaned text
        """
        # Normalize line endings and whitespace
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')
        
        # Remove excessive whitespace
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(line for line in lines if line)
        
        # Fix common OCR errors
        # l (letter L) → 1 (before units/numbers)
        text = re.sub(r'\bl(\d)', r'1\1', text, flags=re.IGNORECASE)
        
        # O (letter O) → 0 (before units/numbers)
        text = re.sub(r'\bO(\d)', r'0\1', text, flags=re.IGNORECASE)
        
        # Normalize decimal separators (European comma to period)
        text = re.sub(r'(\d),(\d)', r'\1.\2', text)
        
        # Remove extra punctuation between numbers
        text = re.sub(r'(\d)\s*[-–—]\s*(\d)', r'\1 \2', text)
        
        # Fix spacing around units
        text = re.sub(r'(\d)\s+([a-z]+)', r'\1\2', text, flags=re.IGNORECASE)
        
        # Normalize common unit abbreviations
        unit_fixes = {
            r'\bkcai\b': 'kcal',
            r'\bkj\b': 'kJ',
            r'\bmg\b': 'mg',
            r'\bg\b': 'g',
        }
        for pattern, replacement in unit_fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # One more whitespace cleanup
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def _extract_serving_size(text: str) -> Optional[float]:
        """
        Extract serving size in grams from OCR text.
        
        Looks for patterns like:
        - Serving size: 30g
        - Per serving (30g)
        - One serving = 30g
        
        Args:
            text: Cleaned OCR text
        
        Returns:
            Serving size in grams, or None if not found
        """
        # Normalize decimal separators
        normalized_text = text.replace(',', '.')
        
        patterns = [
            r'serving\s+size[:\s]+([0-9.]+)\s*g',
            r'per\s+serving\s+\(([0-9.]+)\s*g\)',
            r'one\s+serving\s*=?\s*([0-9.]+)\s*g',
            r'serving[:\s]+([0-9.]+)\s*g',
            r'\(([0-9.]+)\s*g\)\s+serving',
            r'serving\s+([0-9.]+)\s*g',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, normalized_text, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    serving_size = float(match.group(1))
                    # Sanity check: serving size should be between 5g and 500g
                    if 5 <= serving_size <= 500:
                        logger.debug(f"  serving_size: {serving_size}g")
                        return serving_size
                except (ValueError, AttributeError):
                    continue
        
        return None
    
    @staticmethod
    def _extract_nutrition_values(text: str) -> Dict[str, Optional[float]]:
        """
        Extract nutrition values from OCR text.
        
        Uses flexible regex patterns to find common nutrients.
        Handles various label formats and OCR errors.
        
        Args:
            text: Cleaned OCR text
        
        Returns:
            Dictionary of extracted nutrient values
        """
        nutrition_values = {
            'energy_kcal': None,
            'energy_kj': None,
            'fat_g': None,
            'saturated_fat_g': None,
            'trans_fat_g': None,
            'carbohydrates_g': None,
            'sugars_g': None,
            'fiber_g': None,
            'protein_g': None,
            'sodium_mg': None,
            'salt_g': None,
        }
        
        # Normalize decimal separators for matching
        normalized_text = text.replace(',', '.')
        
        # More comprehensive patterns with multiple fallbacks
        patterns = {
            'energy_kcal': [
                r'(?:calories?|energy)[:\s]+(\d+(?:\.\d+)?)\s*(?:kcal|cal)[\s]*(?:\n|$)',
                r'calories?(?:\s+[a-z]+)*[:\s]+(\d+(?:\.\d+)?)',
                r'(?:total\s+)?(?:calories?|energy)[:\s]*(\d+(?:\.\d+)?)',
                r'calories?\s+[•\-\s]*(\d+(?:\.\d+)?)',
            ],
            'energy_kj': [
                r'(?:energy|kj)[:\s]+(\d+(?:\.\d+)?)\s*kj',
                r'kj[:\s]*(\d+(?:\.\d+)?)',
            ],
            'fat_g': [
                r'(?:total\s+)?fat[:\s]*(\d+(?:\.\d+)?)\s*g',
                r'fat\s+[•\-\s]*(\d+(?:\.\d+)?)',
                r'lipid[:\s]+(\d+(?:\.\d+)?)',
            ],
            'saturated_fat_g': [
                r'(?:of\s+which\s+)?(?:saturated|sat\.?)\s+fat[:\s]*(\d+(?:\.\d+)?)',
                r'saturated\s+fat\s+[•\-\s]*(\d+(?:\.\d+)?)',
                r'saturated[:\s]*(\d+(?:\.\d+)?)',
            ],
            'trans_fat_g': [
                r'trans(?:\s+fat)?[:\s]*(\d+(?:\.\d+)?)',
            ],
            'carbohydrates_g': [
                r'(?:total\s+)?carbohydrate[s]?[:\s]*(\d+(?:\.\d+)?)\s*g',
                r'carbohydrate[s]?\s+[•\-\s]*(\d+(?:\.\d+)?)',
                r'carbs[:\s]*(\d+(?:\.\d+)?)',
            ],
            'sugars_g': [
                r'(?:of\s+which\s+)?sugar[s]?[:\s]*(\d+(?:\.\d+)?)',
                r'sugar[s]?\s+[•\-\s]*(\d+(?:\.\d+)?)',
            ],
            'fiber_g': [
                r'(?:dietary\s+)?fiber[:\s]*(\d+(?:\.\d+)?)',
                r'fiber\s+[•\-\s]*(\d+(?:\.\d+)?)',
            ],
            'protein_g': [
                r'protein[:\s]*(\d+(?:\.\d+)?)\s*g',
                r'protein\s+[•\-\s]*(\d+(?:\.\d+)?)',
            ],
            'sodium_mg': [
                r'sodium[:\s]*(\d+(?:\.\d+)?)\s*(?:mg|milligrams)',
                r'sodium\s+[•\-\s]*(\d+(?:\.\d+)?)',
            ],
            'salt_g': [
                r'salt[:\s]*(\d+(?:\.\d+)?)\s*g',
                r'salt\s+[•\-\s]*(\d+(?:\.\d+)?)',
            ],
        }
        
        for nutrient, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, normalized_text, re.IGNORECASE | re.MULTILINE)
                if match:
                    try:
                        value = float(match.group(1))
                        
                        # Sanity check for value ranges
                        if nutrient.endswith('_g'):
                            if 0 <= value <= 100:
                                nutrition_values[nutrient] = value
                                logger.debug(f"  {nutrient}: {value}g")
                                break
                        elif nutrient.endswith('_mg'):
                            if 0 <= value <= 10000:
                                nutrition_values[nutrient] = value
                                logger.debug(f"  {nutrient}: {value}mg")
                                break
                        elif nutrient.endswith('_kcal'):
                            if 0 <= value <= 1000:
                                nutrition_values[nutrient] = value
                                logger.debug(f"  {nutrient}: {value}kcal")
                                break
                        elif nutrient.endswith('_kj'):
                            if 0 <= value <= 5000:
                                nutrition_values[nutrient] = value
                                logger.debug(f"  {nutrient}: {value}kJ")
                                break
                    except (ValueError, AttributeError):
                        continue
        
        return nutrition_values
    
    @staticmethod
    def _extract_ingredients(text: str) -> str:
        """
        Extract ingredients section from text.
        
        Handles OCR errors and various label formats.
        
        Args:
            text: Cleaned OCR text
        
        Returns:
            Ingredients string
        """
        # Multiple patterns to handle OCR errors and different formats
        patterns = [
            # Handle typos: INGREOIENTS, INGREDEINTS, INGREDENTS, etc.
            r'(?:ingre[od]ient|ingred[ie]ent)[s]?[:\s]*(.+?)(?=\n(?:allergen|\*|contains|may contain|nutrition)|contains:|\nallergen|$)',
            # Standard format
            r'ingredients?[:\s]+(.+?)(?=\nallergen|\nnutrition|\ncontains allergy|contains:|$)',
            # Without colon
            r'ingredients?[\s]+(.+?)(?=\nallergen|nutrition|contains allergy)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                ingredients = match.group(1).strip()
                # Clean up the text
                ingredients = ' '.join(ingredients.split())
                # Limit length
                if len(ingredients) > 1000:
                    ingredients = ingredients[:1000]
                if ingredients:
                    logger.debug(f"Ingredients found: {len(ingredients)} chars")
                    return ingredients
        
        return ""
    
    @staticmethod
    def _extract_allergens(text: str) -> str:
        """
        Extract allergen information from text.
        
        Handles various label formats and typos.
        
        Args:
            text: Cleaned OCR text
        
        Returns:
            Allergen string
        """
        patterns = [
            # "Allergens:" or "May contain:"
            r'(?:allergen[s]?|may\s+contain|contains)[:\s]+(.+?)(?=\nnutrition|contains:|$)',
            # Alternative format
            r'(?:allergen[s]?)[:\s]*(.+?)(?=\nnutrition|$)',
            # Special case: "Contains:" followed by allergens
            r'contains[:\s]+([^.\n]+)(?:[\s]*[.])?(?=\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                allergens = match.group(1).strip()
                # Clean up
                allergens = ' '.join(allergens.split())
                # Limit length
                if len(allergens) > 500:
                    allergens = allergens[:500]
                if allergens and len(allergens) > 3:  # Avoid single-word false positives
                    logger.debug(f"Allergens found: {len(allergens)} chars")
                    return allergens
        
        return ""


# Global OCR instance
_ocr_engine = None


def get_ocr_engine(tesseract_path: Optional[str] = None) -> NutritionOCR:
    """
    Get or create OCR engine instance (singleton).
    
    Args:
        tesseract_path: Ignored (for backward compatibility)
    
    Returns:
        NutritionOCR instance
    """
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = NutritionOCR()
    return _ocr_engine
