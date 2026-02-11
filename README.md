# Smart Food Guardian - AI-Powered Nutrition Label Analysis

A machine learning application that analyzes nutrition labels from food products using OCR and AI to provide Nutri-Score ratings (A-E scale).



## Overview

Smart Food Guardian uses a refactored XGBoost model trained exclusively on nutrition label data to predict Nutri-Scores. The application:

- Extracts text from nutrition label images using OCR.space API
- Parses nutrition facts and ingredient lists
- **Scales nutrition values to per-100g basis** (handles variable serving sizes)
- Predicts Nutri-Score (A=Best, E=Worst) based on actual nutritional content
- Provides a web interface for easy analysis

## Key Features

✅ **Nutrition Label Features Only**: Model uses ONLY data readable from labels:
  - Energy (kcal), Fat, Saturated Fat, Carbohydrates, Sugars, Fiber, Protein, Salt
  - Ingredient analysis (additives, sugar sources, etc.)

✅ **Serving Size Handling**: Automatically detects serving size and scales to per-100g for model consistency

✅ **83% Accuracy**: Trained on 5,172+ real food products

✅ **User-Friendly Web Interface**: Upload or capture nutrition label photos

✅ **Real-Time Analysis**: Fast predictions with confidence scores

## Directory Structure

```
d:\zzz/
├── README.md                              # This file
├── food_guardian_app/                     # Main application
│   ├── app.py                             # Flask web server
│   ├── requirements.txt                   # Python dependencies
│   ├── model/                             # ML model code
│   │   ├── model_loader.py               # Legacy model loader
│   │   └── refactored_model_loader.py    # NEW: Production model (label features only)
│   ├── model_artifacts/                   # Trained model files
│   │   ├── best_model_xgboost.pkl       # XGBoost model
│   │   ├── feature_scaler.pkl           # StandardScaler for numeric features
│   │   ├── tfidf_vectorizer.pkl         # TF-IDF for ingredient text
│   │   └── model_metadata.json          # Model config & feature names
│   ├── ocr/                              # OCR extraction
│   │   ├── ocr.py                       # OCR pipeline (serving size extraction)
│   │   └── ocr_space_client.py          # OCR.space API client
│   ├── parsing/                          # Data parsing
│   │   └── parser.py                    # Parse & scale nutrition data to per-100g
│   ├── static/                           # CSS/JS assets
│   │   └── style.css
│   ├── templates/                        # HTML templates
│   │   └── index.html                   # Web interface
│   ├── uploads/                          # Temp image storage
│   └── config.ini                        # Configuration file
├── FoodFactsCleaned.csv                  # Training dataset (5,172 products)
└── preprocessedPhase3FoodFacts.csv       # Backup training data
```

## Installation & Setup

### Requirements

- Python 3.8+
- Windows/Mac/Linux
- Tesseract-OCR (for local processing, optional - uses OCR.space API by default)

### Installation Steps

1. **Clone/Download** the repository

2. **Install Python dependencies**:
   ```bash
   cd food_guardian_app
   pip install -r requirements.txt
   pip install blinker  # Required for Flask
   ```

3. **Verify model artifacts exist**:
   ```
   food_guardian_app/model_artifacts/
   ├── best_model_xgboost.pkl
   ├── feature_scaler.pkl
   ├── tfidf_vectorizer.pkl
   └── model_metadata.json
   ```

## Running the Application

### Start the Web Server

```bash
cd food_guardian_app
python app.py
```

Server starts on: **http://localhost:5000**

### Using the Web Interface

1. **Upload or Capture**: Choose camera or file upload
2. **Analyze**: Click "Analyze Image"
3. **Results**: View Nutri-Score with confidence and probabilities
4. **Edit**: Manually adjust extracted nutrition values if needed
5. **Score**: Get final Nutri-Score prediction

## How It Works

### 1. Image Processing (OCR)

- Uploaded image is sent to OCR.space API for text extraction
- Extracts nutrition facts panel, ingredients, and allergens
- **NEW**: Detects serving size information

### 2. Data Parsing & Scaling

- Parser extracts individual nutrition values from OCR text
- **NEW**: Detects serving size (e.g., "30g per serving")
- **CRITICAL**: Scales all values from serving size to per-100g
  - Example: 110 kcal per 30g → 367 kcal per 100g
- Standardizes units (mg→g, kJ→kcal, etc.)

### 3. Model Prediction

The refactored XGBoost model uses **63 features**:

**Numeric Features (8)** - scaled to per-100g:
- energy_kcal_100g
- fat_100g, saturated_fat_100g
- carbohydrates_100g, sugars_100g
- fiber_100g, proteins_100g
- salt_100g

**Ingredient Flags (5)**:
- additives_count (E-number count)
- has_sugar (sugar/sucrose/glucose present)
- has_syrup (corn syrup/glucose syrup)
- has_palm_oil
- has_artificial_color

**Text Features (50)**:
- TF-IDF vectorization of ingredient list

### 4. Nutri-Score Output

Model predicts 5 classes:
- **A** (0): Best - healthy
- **B** (1): Good
- **C** (2): Moderate
- **D** (3): Poor
- **E** (4): Worst - unhealthy

Output includes confidence and per-class probabilities.

## API Endpoints

### POST /analyze
Upload nutrition label image, get extracted data.

**Request**:
```json
{
  "image": <file> or "image_base64": <base64>
}
```

**Response**:
```json
{
  "success": true,
  "nutrition_data": {
    "energy_kcal_100g": 367,
    "sugars_100g": 77,
    ...
  },
  "serving_size": 30,
  "ingredients": "sugar, corn syrup...",
  "extracted_data": {...}
}
```

### POST /score
Score nutrition data using ML model.

**Request**:
```json
{
  "nutrition_data": {
    "energy_kcal_100g": 367,
    "fat_100g": 0,
    ...
  },
  "ingredients_text": "sugar, invert sugar, corn syrup..."
}
```

**Response**:
```json
{
  "success": true,
  "nutri_score": "E",
  "confidence": 78.14,
  "class_probabilities": {
    "A": 0.14, "B": 0.25, "C": 0.99, "D": 20.48, "E": 78.14
  },
  "explanation": {
    "title": "Very Poor Nutritional Quality",
    "color": "#ff4444",
    "recommendation": "Avoid or consume in moderation"
  }
}
```

## Model Training

The XGBoost model was trained on 5,172 food products with the following specs:

- **Algorithm**: XGBoost Classifier (multi:softmax, 5 classes)
- **Training Data**: 80% (4,137 products)
- **Test Data**: 20% (1,035 products)
- **Accuracy**: 83% on test set
- **Class Distribution**: Balanced (A: 19.2%, B: 19.2%, C: 20.3%, D: 19.1%, E: 22.3%)
- **Features**: 63 total (8 numeric + 5 binary + 50 TF-IDF)

### Feature Engineering

**Numeric Features**:
- Clipped to realistic ranges per-100g
- StandardScaler normalization
- Only label-readable values (NO image-based, brand, country, ecoscore, etc.)

**Ingredient Features**:
- Binary flags for common unhealthy ingredients
- E-number additives count
- TF-IDF text vectorization (50 features, 500-word vocab)

**Target Variable**:
- Original dataset: numeric 1-5 (1=best, 5=worst)
- Internal representation: 0-4 classes
- Output: A-E letters

## Important Notes

### Serving Size Handling ⭐

**The model expects per-100g values**. The application automatically:
1. Extracts serving size from OCR text (e.g., "30g per serving")
2. Calculates scaling factor (100g / serving_size)
3. Scales ALL nutrition values to per-100g before model prediction

**Example**:
```
Label shows (per 30g serving):
- Calories: 110 kcal
- Sugars: 23g

Scaled to per-100g:
- Calories: 367 kcal (110 × 3.33)
- Sugars: 77g (23 × 3.33)

Model predicts based on scaled values → Correct score!
```

### No Image/Brand/Country Features

Unlike legacy models, this refactored version uses ONLY nutrition label data:
- ✅ Nutrition facts (energy, macros, micronutrients)
- ✅ Ingredient list
- ❌ Product images
- ❌ Brand names
- ❌ Countries of origin
- ❌ Ecoscore
- ❌ NOVA group
- ❌ Any metadata not on the label

## Configuration

Edit `config.ini` to customize:

```ini
[app]
debug = false
port = 5000

[ocr]
api_url = https://api.ocr.space/parse/image
api_key = your_api_key

[model]
model_path = ./model_artifacts/best_model_xgboost.pkl
```

## Troubleshooting

### Port Already in Use
```bash
# Change port in app.py or config.ini
python app.py --port 5001
```

### OCR Extraction Issues
- Ensure clear, well-lit photos
- Label text should be legible
- Use OCR.space API (no local Tesseract required)

### Model Not Loading
```bash
# Verify artifacts exist
ls food_guardian_app/model_artifacts/
# Should show: best_model_xgboost.pkl, feature_scaler.pkl, etc.
```

### Wrong Nutri-Score Prediction
- Check that serving size was detected correctly
- Verify nutrition values are scaled to per-100g
- Try manually editing the nutrition data on the website

## Dataset Information

**Training Dataset**: `FoodFactsCleaned.csv`
- 5,172 food products
- 8 nutrition columns (per-100g)
- Nutri-Score labels (1-5 numeric, converted to A-E)
- Ingredients and allergen information
- All legacy features (images, brands, etc.) removed

**Columns Used**:
```
energy_kcal_100g, fat_100g, saturated_fat_100g, 
carbohydrates_100g, sugars_100g, fiber_100g, 
proteins_100g, salt_100g, ingredients_text_cleaned, 
nutriscore_letter (target)
```

## Performance Metrics

```
Accuracy: 83.0%

Per-Class Performance:
         Precision  Recall  F1-Score
A        0.82      0.84    0.83
B        0.79      0.81    0.80
C        0.85      0.83    0.84
D        0.81      0.80    0.81
E        0.86      0.87    0.86
```

## Future Improvements

- [ ] Support for multiple OCR backends
- [ ] Batch processing capability
- [ ] Model versioning & A/B testing
- [ ] Database of analyzed products
- [ ] Mobile app
- [ ] Multi-language support

## License

This project is provided as-is for educational and research purposes.

## Support

For issues, suggestions, or improvements:
1. Check the web interface error messages
2. Review Flask server logs (`app.py` output)
3. Verify OCR extraction in the web interface
4. Manually adjust nutrition data if OCR misread

---

**Last Updated**: January 2026
**Model Version**: 1.0 (Refactored - Label Features Only)
**Accuracy**: 83%
