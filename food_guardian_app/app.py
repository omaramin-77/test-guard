"""
Smart Food Guardian - Flask Web Application
============================================
A production-ready web app for analyzing nutrition labels using OCR and machine learning.

Features:
- Camera capture and image upload
- Automatic text extraction from nutrition labels
- ML-based nutrition scoring
- User editable extraction results
"""

import os
import json
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import logging

# Import custom modules
from model.refactored_model_loader import get_model_loader
from ocr.ocr import get_ocr_engine
from parsing.parser import get_parser

# ============================================================================
# Configuration
# ============================================================================

# Flask app setup
app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')

# Configuration
UPLOAD_FOLDER = Path('uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['JSON_SORT_KEYS'] = False

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Helper Functions
# ============================================================================

def allowed_file(filename: str) -> bool:
    """Check if uploaded file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_file(file) -> tuple:
    """
    Save uploaded file safely.
    
    Returns:
        (success: bool, filepath: str, error: str)
    """
    if not file or file.filename == '':
        return False, '', 'No file selected'
    
    if not allowed_file(file.filename):
        return False, '', f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
    
    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"File saved: {filepath}")
        return True, filepath, None
        
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        return False, '', f'Error saving file: {str(e)}'


def format_nutrition_for_display(nutrition: dict) -> dict:
    """Format nutrition data for clean HTML display."""
    display = {}
    
    # Core nutrients
    nutrients = [
        ('energy_kcal_100g', 'Energy', 'kcal'),
        ('fat_100g', 'Total Fat', 'g'),
        ('saturated_fat_100g', 'Saturated Fat', 'g'),
        ('carbohydrates_100g', 'Carbohydrates', 'g'),
        ('sugars_100g', 'Sugars', 'g'),
        ('fiber_100g', 'Dietary Fiber', 'g'),
        ('proteins_100g', 'Protein', 'g'),
        ('salt_100g', 'Salt', 'g'),
    ]
    
    for key, name, unit in nutrients:
        if key in nutrition:
            value = nutrition[key]
            if value is not None:
                display[name] = f"{value:.2f} {unit}"
    
    return display


def get_nutri_score_explanation(score_letter: str) -> dict:
    """Get explanation and color for Nutri-Score."""
    explanations = {
        'A': {
            'color': '#2ecc71',
            'title': 'Excellent Nutrition',
            'description': 'This product has a very good nutritional quality. Enjoy it!',
            'recommendation': 'Highly recommended for regular consumption.'
        },
        'B': {
            'color': '#a8d84d',
            'title': 'Good Nutrition',
            'description': 'This product has good nutritional quality.',
            'recommendation': 'Suitable for regular consumption. A solid choice.'
        },
        'C': {
            'color': '#ffd633',
            'title': 'Moderate Nutrition',
            'description': 'This product has moderate nutritional quality.',
            'recommendation': 'Can be consumed, but better options may be available.'
        },
        'D': {
            'color': '#ff9933',
            'title': 'Poor Nutrition',
            'description': 'This product has lower nutritional quality.',
            'recommendation': 'Limit consumption. Consider healthier alternatives.'
        },
        'E': {
            'color': '#ff4444',
            'title': 'Very Poor Nutrition',
            'description': 'This product has poor nutritional quality.',
            'recommendation': 'Minimize consumption. Choose healthier options.'
        }
    }
    
    return explanations.get(score_letter, explanations['C'])


# ============================================================================
# Route: Home / Index
# ============================================================================

@app.route('/')
def index():
    """
    Home page with camera/upload interface.
    
    GET /
        Returns: HTML form for image upload or camera capture
    """
    logger.info("GET /")
    return render_template('index.html')


# ============================================================================
# Route: Analyze Image
# ============================================================================

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint.
    
    POST /analyze
        Form data:
            - image: Image file (from camera or upload)
            - or image_base64: Base64 encoded image from camera
        
        Returns: JSON with OCR results and parsed nutrition data
    """
    logger.info("POST /analyze")
    
    try:
        # Initialize components
        ocr_engine = get_ocr_engine()
        parser = get_parser()
        
        # Handle image upload
        image_path = None
        
        if 'image' in request.files:
            # File upload
            file = request.files['image']
            success, image_path, error = save_uploaded_file(file)
            
            if not success:
                return jsonify({'success': False, 'error': error}), 400
        
        elif 'image_base64' in request.form:
            # Camera capture (base64)
            import base64
            
            try:
                base64_str = request.form['image_base64']
                # Remove data:image/jpeg;base64, prefix if present
                if ',' in base64_str:
                    base64_str = base64_str.split(',')[1]
                
                image_data = base64.b64decode(base64_str)
                
                # Save as file
                filename = secure_filename(
                    f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                )
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                
                logger.info(f"Camera image saved: {image_path}")
                
            except Exception as e:
                logger.error(f"Error processing base64 image: {e}")
                return jsonify({'success': False, 'error': f'Error processing image: {str(e)}'}), 400
        
        else:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({'success': False, 'error': 'Image file not found'}), 400
        
        # ====================================================================
        # OCR Extraction
        # ====================================================================
        logger.info(f"Starting OCR on: {image_path}")
        try:
            ocr_result = ocr_engine.full_extraction(image_path)
        except Exception as ocr_error:
            logger.error(f"OCR extraction error: {ocr_error}", exc_info=True)
            return jsonify({
                'success': False,
                'error': f'OCR Error: {str(ocr_error)}. Make sure Tesseract-OCR is installed.'
            }), 400
        
        if not ocr_result.get('success', False):
            error_msg = ocr_result.get('error', 'Unknown OCR error')
            logger.warning(f"OCR failed: {error_msg}")
            return jsonify({
                'success': False,
                'error': f"OCR extraction failed: {error_msg}. Ensure Tesseract-OCR is installed."
            }), 400
        
        # Check if OCR extracted any text
        if not ocr_result.get('raw_text', '').strip():
            logger.warning("OCR extracted no text from image")
            return jsonify({
                'success': False,
                'error': 'No text could be extracted from the image. Try uploading a clearer image with better lighting.'
            }), 400
        
        logger.info("OCR extraction successful")
        
        # ====================================================================
        # Parsing
        # ====================================================================
        logger.info("Starting parsing")
        parse_result = parser.parse_ocr_extraction(ocr_result)
        
        if not parse_result.get('success', False):
            logger.warning("Parsing failed")
            return jsonify({'success': False, 'error': 'Parsing failed'}), 400
        
        logger.info("Parsing successful")
        
        # ====================================================================
        # Format Response
        # ====================================================================
        response = {
            'success': True,
            'image_path': image_path,
            'raw_text': ocr_result.get('raw_text', ''),
            'cleaned_text': ocr_result.get('cleaned_text', ''),
            'extracted_data': {
                'ingredients': parse_result.get('ingredients', ''),
                'allergens': parse_result.get('allergens', ''),
                'nutrition_values': format_nutrition_for_display(
                    parse_result.get('standardized_nutrition', {})
                ),
            },
            'nutrition_data': parse_result.get('standardized_nutrition', {}),
            'nutrient_levels': parse_result.get('nutrient_levels', {}),
            'validation': {
                k: v[1] for k, v in parse_result.get('validation', {}).items()
            },
            'ready_for_model': True
        }
        
        logger.info("Analysis response prepared")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error in /analyze: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


# ============================================================================
# Route: Score Nutrition
# ============================================================================

@app.route('/score', methods=['POST'])
def score():
    """
    Score nutrition data using the refactored XGBoost model.
    
    POST /score
        JSON body:
            {
                'nutrition_data': {
                    'energy_kcal_100g': float,
                    'fat_100g': float,
                    'saturated_fat_100g': float,
                    'carbohydrates_100g': float,
                    'sugars_100g': float,
                    'fiber_100g': float,
                    'proteins_100g': float,
                    'salt_100g': float
                },
                'ingredients_text': string
            }
        
        Returns: JSON with Nutri-Score and confidence
    """
    logger.info("POST /score")
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
        
        # Get model loader (NEW: refactored version)
        model_loader = get_model_loader(model_dir='./model_artifacts')
        
        # Extract required fields
        nutrition_data = data.get('nutrition_data', {})
        ingredients_text = data.get('ingredients_text', '')
        
        if not nutrition_data:
            return jsonify({'success': False, 'error': 'nutrition_data is required'}), 400
        
        logger.info(f"Scoring with {len(nutrition_data)} nutrition fields")
        
        try:
            # Make prediction (NEW: simplified API)
            prediction = model_loader.predict(nutrition_data, ingredients_text)
            
        except ValueError as ve:
            # Catches forbidden feature errors
            logger.error(f"Forbidden feature error: {ve}")
            return jsonify({
                'success': False,
                'error': str(ve)
            }), 400
        
        # Get Nutri-Score explanation
        nutri_score = prediction['nutri_score']
        explanation = get_nutri_score_explanation(nutri_score)
        
        # Build response
        response = {
            'success': True,
            'nutri_score': nutri_score,
            'confidence': round(prediction['confidence'] * 100, 2),
            'class_probabilities': {
                score: round(prob * 100, 2)
                for score, prob in prediction['predictions'].items()
            },
            'explanation': explanation,
            'recommendation': explanation['recommendation']
        }
        
        logger.info(f"Prediction: {nutri_score} (confidence: {response['confidence']}%)")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error in /score: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Scoring error: {str(e)}'
        }), 500


# ============================================================================
# Route: Edit Nutrition Data
# ============================================================================

@app.route('/edit-nutrition', methods=['POST'])
def edit_nutrition():
    """
    Allow user to edit extracted nutrition values before scoring.
    
    POST /edit-nutrition
        JSON body:
            {
                'nutrition_data': {edited values},
                'ingredients': string,
                'allergens': string
            }
        
        Returns: Updated nutrition data and validation
    """
    logger.info("POST /edit-nutrition")
    
    try:
        data = request.get_json()
        parser_instance = get_parser()
        
        nutrition_data = data.get('nutrition_data', {})
        
        # Apply edits
        updated = parser_instance.apply_user_edits(nutrition_data)
        
        # Get validation
        validation = parser_instance.validate_nutrition_values(nutrition_data)
        
        response = {
            'success': True,
            'updated_nutrition': updated,
            'formatted_nutrition': format_nutrition_for_display(nutrition_data),
            'validation': {k: v[1] for k, v in validation.items()}
        }
        
        logger.info("Nutrition data edited successfully")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error in /edit-nutrition: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Edit error: {str(e)}'
        }), 500


# ============================================================================
# Route: Health Check
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }), 200


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({
        'success': False,
        'error': f'File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.1f} MB'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# ============================================================================
# Initialization & Main
# ============================================================================

def initialize_app():
    """Initialize app components."""
    logger.info("=" * 80)
    logger.info("Smart Food Guardian - REFACTORED (Nutrition Label Features Only)")
    logger.info("=" * 80)
    
    # Initialize refactored model loader
    try:
        model_loader = get_model_loader(model_dir='./model_artifacts')
        logger.info("✓ Refactored model loader initialized")
        logger.info(f"✓ Model accuracy on test set: {model_loader.metadata.get('model_accuracy', 'N/A')}")
        logger.info(f"✓ Model uses ONLY nutrition label features (no images, brands, countries, etc.)")
    except Exception as e:
        logger.error(f"❌ Failed to load refactored model: {e}")
        logger.info("Please run the refactored training notebook first:")
        logger.info("  python refactor_ml_pipeline.ipynb")
    
    # Initialize OCR engine with Windows Tesseract path if on Windows
    import platform
    tesseract_path = None
    if platform.system() == 'Windows':
        # Try common Windows installation paths
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        for path in possible_paths:
            if os.path.exists(path):
                tesseract_path = path
                logger.info(f"✓ Found Tesseract at: {path}")
                break
        if not tesseract_path:
            logger.warning("Tesseract not found at standard Windows paths. Please ensure Tesseract-OCR is installed.")
            logger.info("Download from: https://github.com/UB-Mannheim/tesseract/wiki")
    
    ocr_engine = get_ocr_engine(tesseract_path=tesseract_path)
    logger.info("✓ OCR Engine initialized")
    
    # Initialize parser
    parser = get_parser()
    logger.info("✓ Parser initialized")
    
    logger.info("=" * 80)
    logger.info("Smart Food Guardian - Ready")
    logger.info("=" * 80)


if __name__ == '__main__':
    # Initialize components
    initialize_app()
    
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    is_production = os.environ.get('ENVIRONMENT') == 'production'
    
    if is_production:
        logger.info("Starting Flask production server...")
    else:
        logger.info("Starting Flask development server...")
        logger.info("Visit http://localhost:5000 in your browser")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=not is_production,
        use_reloader=False
    )
