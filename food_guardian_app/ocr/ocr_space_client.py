""""
OCR.space API Client
====================
Provides robust OCR through the free OCR.space API.

This module:
- Sends images to OCR.space for processing
- Handles API responses
- Manages errors gracefully
- Returns structured OCR output

OCR.space provides free OCR without authentication on free tier.
Focuses on printed text recognition (nutrition labels).
"""

import requests
import logging
from typing import Dict, Optional, List
from pathlib import Path
import time
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class OCRSpaceClient:
    """
    Client for OCR.space free OCR API.
    
    Features:
    - Free tier usage (no API key required)
    - Supports English printed text
    - Returns structured output
    - Handles errors gracefully
    - Includes retry logic
    """
    
    # OCR.space free API endpoint
    API_ENDPOINT = "https://api.ocr.space/Parse/Image"
    API_ENDPOINT_PAID = "https://api.ocr.space/Parse/Image"  # Same endpoint works for both
    
    # Supported languages
    LANGUAGE = "eng"  # English
    
    # API parameters
    REQUEST_TIMEOUT = 60  # seconds
    MAX_RETRIES = 2
    RETRY_DELAY = 2  # seconds
    
    def __init__(self):
        """Initialize OCR client."""
        logger.info("OCR.space API client initialized")
        self.session = requests.Session()
    
    def extract_text_from_image(self, image_path: str) -> Dict[str, any]:
        """
        Extract text from image using OCR.space API.
        
        Args:
            image_path: Path to image file (JPG, PNG, BMP, TIFF, GIF)
        
        Returns:
            Dictionary with:
            {
                'success': bool,
                'raw_text': str (complete extracted text),
                'is_error': bool,
                'error_message': str or None,
                'confidence': float (0-100, estimated),
                'lines': List[str] (text by line),
            }
        """
        logger.info(f"Starting OCR for: {image_path}")
        
        # Validate file exists
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            error_msg = f"Image file not found: {image_path}"
            logger.error(error_msg)
            return {
                'success': False,
                'raw_text': '',
                'is_error': True,
                'error_message': error_msg,
                'confidence': 0,
                'lines': []
            }
        
        # Validate file size (OCR.space has limits)
        file_size_mb = image_path_obj.stat().st_size / (1024 * 1024)
        if file_size_mb > 50:  # 50 MB limit
            error_msg = f"Image too large: {file_size_mb:.1f} MB (max 50 MB)"
            logger.error(error_msg)
            return {
                'success': False,
                'raw_text': '',
                'is_error': True,
                'error_message': error_msg,
                'confidence': 0,
                'lines': []
            }
        
        # Try OCR with retries
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                logger.info(f"OCR attempt {attempt + 1}/{self.MAX_RETRIES + 1}")
                result = self._call_ocr_space_api(image_path)
                
                if result['success']:
                    logger.info(f"OCR successful: {len(result['raw_text'])} chars extracted")
                    return result
                elif attempt < self.MAX_RETRIES:
                    logger.warning(f"OCR failed (attempt {attempt + 1}), retrying in {self.RETRY_DELAY}s...")
                    time.sleep(self.RETRY_DELAY)
                else:
                    logger.error("OCR failed after all retries")
                    return result
                    
            except Exception as e:
                logger.error(f"Exception during OCR (attempt {attempt + 1}): {e}", exc_info=True)
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY)
                else:
                    return {
                        'success': False,
                        'raw_text': '',
                        'is_error': True,
                        'error_message': f'OCR service error: {str(e)}',
                        'confidence': 0,
                        'lines': []
                    }
        
        return {
            'success': False,
            'raw_text': '',
            'is_error': True,
            'error_message': 'OCR failed after all retries',
            'confidence': 0,
            'lines': []
        }
    
    def _call_ocr_space_api(self, image_path: str) -> Dict[str, any]:
        """
        Internal: Call OCR.space API with image file.
        
        Args:
            image_path: Path to image file
        
        Returns:
            API response as structured dictionary
        """
        try:
            with open(image_path, 'rb') as image_file:
                # Read file as binary
                image_data = image_file.read()
                
                # Get filename for reference
                filename = Path(image_path).name
                
                # Prepare files parameter - OCR.space expects 'filename' parameter
                files = {
                    'filename': (filename, image_data, 'image/png')
                }
                
                # Prepare data parameters
                data = {
                    'language': self.LANGUAGE,
                    'isOverlayRequired': False,  # Don't need overlay image
                }
                
                # Prepare headers with API key if available
                headers = {}
                api_key = os.getenv('OCR_SPACE_API_KEY')
                if api_key:
                    headers['apikey'] = api_key
                    logger.debug(f"Using OCR.space API with paid key")
                else:
                    logger.debug(f"No API key, using free tier (limited requests)")
                
                logger.debug(f"Sending request to OCR.space: {self.API_ENDPOINT}")
                
                # Call OCR.space API
                response = requests.post(
                    self.API_ENDPOINT,
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=self.REQUEST_TIMEOUT
                )
                
                logger.debug(f"API Response Status: {response.status_code}")
                response.raise_for_status()  # Raise on HTTP errors
                api_response = response.json()
                
                # Check API response structure
                if not isinstance(api_response, dict):
                    raise ValueError(f"Unexpected API response type: {type(api_response)}")
                
                # Check for OCR errors
                is_error = api_response.get('IsErroredOnProcessing', False)
                error_message = api_response.get('ErrorMessage', '')
                
                if is_error:
                    logger.warning(f"OCR.space returned error: {error_message}")
                    return {
                        'success': False,
                        'raw_text': '',
                        'is_error': True,
                        'error_message': error_message or 'OCR processing failed',
                        'confidence': 0,
                        'lines': []
                    }
                
                # Extract text from ParsedResults (which is an array of pages)
                raw_text = ''
                lines = []
                results = api_response.get('ParsedResults', [])
                
                if results and len(results) > 0:
                    # Get the first page's parsed text
                    parsed_text = results[0].get('ParsedText', '')
                    if parsed_text:
                        raw_text = parsed_text
                        # Split into lines and clean
                        lines = [line.strip() for line in parsed_text.split('\n') 
                                if line.strip()]
                
                if not raw_text or not raw_text.strip():
                    logger.warning("OCR returned empty text")
                    return {
                        'success': False,
                        'raw_text': '',
                        'is_error': False,
                        'error_message': 'No text extracted from image',
                        'confidence': 0,
                        'lines': []
                    }
                
                # Estimate confidence based on text length and structure
                # (OCR.space free tier doesn't provide per-word confidence)
                confidence = self._estimate_confidence(raw_text, lines)
                
                logger.info(f"OCR success: {len(raw_text)} chars, {len(lines)} lines, confidence={confidence:.0f}%")
                
                return {
                    'success': True,
                    'raw_text': raw_text,
                    'is_error': False,
                    'error_message': None,
                    'confidence': confidence,
                    'lines': lines
                }
                
        except requests.exceptions.Timeout:
            error_msg = f"OCR API timeout after {self.REQUEST_TIMEOUT}s"
            logger.error(error_msg)
            raise
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Failed to connect to OCR API: {e}"
            logger.error(error_msg)
            raise
        except requests.exceptions.HTTPError as e:
            error_msg = f"OCR API HTTP error: {e.response.status_code}"
            logger.error(error_msg)
            raise
        except ValueError as e:
            error_msg = f"Invalid OCR API response: {e}"
            logger.error(error_msg)
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling OCR API: {e}", exc_info=True)
            raise
    
    @staticmethod
    def _estimate_confidence(raw_text: str, lines: List[str]) -> float:
        """
        Estimate OCR confidence based on text characteristics.
        
        Since OCR.space free tier doesn't provide confidence scores,
        we estimate based on:
        - Text length (longer = more confident)
        - Number of lines (structured text is usually clearer)
        - Character distribution (numbers and common words indicate quality)
        
        Args:
            raw_text: Extracted text
            lines: Lines of text
        
        Returns:
            Confidence score 0-100
        """
        confidence = 70  # Base confidence
        
        # More text = better confidence
        if len(raw_text) > 1000:
            confidence += 15
        elif len(raw_text) > 500:
            confidence += 10
        elif len(raw_text) < 100:
            confidence -= 20
        
        # More lines = more structured (better quality)
        if len(lines) > 20:
            confidence += 10
        elif len(lines) < 5:
            confidence -= 10
        
        # Check for common nutrition keywords (indicator of relevant content)
        nutrition_keywords = [
            'nutrition', 'calories', 'fat', 'protein', 'carbohydrates',
            'sugar', 'sodium', 'fiber', 'energy', 'kcal', 'kj',
            'saturated', 'ingredients', 'allergen'
        ]
        keyword_count = sum(1 for kw in nutrition_keywords 
                           if kw.lower() in raw_text.lower())
        confidence += min(15, keyword_count * 2)  # Cap bonus at 15
        
        # Clamp to 0-100
        return max(0, min(100, confidence))


# Global client instance
_ocr_client = None


def get_ocr_client() -> OCRSpaceClient:
    """
    Get or create OCR client instance (singleton).
    
    Returns:
        OCRSpaceClient instance
    """
    global _ocr_client
    if _ocr_client is None:
        _ocr_client = OCRSpaceClient()
    return _ocr_client
