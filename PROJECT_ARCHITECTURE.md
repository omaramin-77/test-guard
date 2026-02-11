# Smart Food Guardian - Complete Technical Overview

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Data Pipeline](#data-pipeline)
4. [OCR System](#ocr-system)
5. [Data Parsing](#data-parsing)
6. [Feature Engineering](#feature-engineering)
7. [Model Training](#model-training)
8. [Flask Application](#flask-application)
9. [Model Loading & Prediction](#model-loading--prediction)
10. [End-to-End Workflow](#end-to-end-workflow)

---

## Project Overview

**Smart Food Guardian** is a machine learning web application that analyzes nutrition labels from food products to predict Nutri-Scores (A=Best, E=Worst).

### Key Principle
The model uses **ONLY data readable from a nutrition label** - no images, brand names, country of origin, or other metadata. This ensures real-world applicability and prevents model bias from non-nutritional factors.

### Technology Stack
- **Backend**: Python, Flask
- **ML Framework**: XGBoost (classification)
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **OCR**: OCR.space API (primary), Tesseract-OCR (optional)
- **Frontend**: HTML5, CSS3, JavaScript

---

## System Architecture

### High-Level Flow

```
User Upload Image
        ↓
   [OCR Module]
        ↓
   [Parser Module]
        ↓
   [Model Loader]
        ↓
   [XGBoost Model]
        ↓
   Nutri-Score Prediction (A-E)
        ↓
   Display Results
```

### Component Diagram

```
┌─────────────────────────────────────────────────┐
│           Flask Web Application (app.py)        │
│  - Handles HTTP requests                        │
│  - Manages file uploads                         │
│  - Coordinates pipeline execution               │
└──────────────┬──────────────────────────────────┘
               │
        ┌──────┴──────┐
        ↓             ↓
    [OCR Module]  [Parser Module]
    ocr/ocr.py    parsing/parser.py
        │             │
        ├─ Extract     ├─ Parse nutrition values
        │  serving     ├─ Scale to per-100g
        │  size        ├─ Extract ingredients
        │              ├─ Calculate features
        ├─ Extract     └─ Validate data
        │  nutrition   
        │  values      
        │
        ├─ Extract
        │  ingredients
        │
        └─ Extract
           allergens
               │
               ↓
    [Model Loading]
    model/refactored_model_loader.py
        │
        ├─ Load trained XGBoost model
        ├─ Load feature scaler
        ├─ Load TF-IDF vectorizer
        └─ Load metadata
               │
               ↓
    [Feature Engineering]
        │
        ├─ Scale numeric features
        ├─ Extract binary ingredients
        └─ Vectorize text (TF-IDF)
               │
               ↓
    [XGBoost Classifier]
        │
        ├─ Input: 63 features
        ├─ Output: 5 classes (A-E)
        └─ Confidence: Probability scores
               │
               ↓
    Nutri-Score (A, B, C, D, or E)
```

---

## Data Pipeline

### 1. Original Dataset

**File**: `preprocessedPhase3FoodFacts.csv`

- **5,898 products** initially
- Nutrition data, product info, ecoscore, images, brands, etc.
- Contains **"nutriscore_letter"** column with values 1-5 (1=best, 5=worst)

### 2. Data Cleaning

**Location**: `COMPLETE_REFACTOR_PIPELINE.py` (lines 60-100)

The training script removes **all forbidden features** and keeps **ONLY nutrition label data**:

```python
FORBIDDEN_FEATURES = {
    'image_path', 'image_160_path', 'has_image',        # Images
    'brand', 'brand_cleaned',                            # Brands
    'countries', 'countries_cleaned',                    # Origin
    'ecoscore_grade', 'ecoscore_score',                 # Ecoscore
    'nova_group', 'additives', 'additives_cleaned',    # Non-label data
    'url', 'product_id', 'packaging',                   # Metadata
    # ... 30+ more forbidden features
}

REQUIRED_FEATURES = {
    'energy_kcal_100g',
    'fat_100g', 'saturated_fat_100g',
    'carbohydrates_100g', 'sugars_100g', 'fiber_100g',
    'proteins_100g', 'salt_100g',
    'ingredients_text_cleaned',
    'nutriscore_letter'  # Target
}
```

**Result**: **5,172 clean products** with complete nutrition data

### 3. Data Types

Each product contains:

| Field | Type | Example | Source |
|-------|------|---------|--------|
| `energy_kcal_100g` | float | 366.7 | Nutrition label |
| `fat_100g` | float | 0.0 | Nutrition label |
| `sugars_100g` | float | 77.0 | Nutrition label (per 100g) |
| `fiber_100g` | float | 0.0 | Nutrition label |
| `proteins_100g` | float | 0.0 | Nutrition label |
| `salt_100g` | float | 0.083 | Nutrition label |
| `ingredients_text_cleaned` | string | "SUGAR, CORN SYRUP..." | Ingredients list |
| `nutriscore_letter` | int (1-5) | 5 | Target variable |

---

## OCR System

### What is OCR?

**OCR = Optical Character Recognition**

OCR converts images of text into machine-readable text. Smart Food Guardian reads nutrition label images and extracts the text automatically.

### OCR in This Project

**Location**: `food_guardian_app/ocr/`

#### Two Implementations:

1. **OCR.space API** (Primary)
   - Cloud-based service
   - Reliable, well-maintained
   - Free tier available
   - No local installation needed
   - Works offline: ❌ (requires internet)

2. **Tesseract-OCR** (Optional Fallback)
   - Local installation
   - Free, open-source
   - Slower but offline-capable
   - Windows: `C:\Program Files\Tesseract-OCR\tesseract.exe`

### How OCR Works in the Pipeline

```
User uploads image (nutrition label)
            ↓
        [OCR.space API]
            ↓
     (or Tesseract-OCR)
            ↓
    Extract text from image
            ↓
    Clean & normalize text
            ↓
Return structured text
```

### Code: OCR Extraction

**File**: `food_guardian_app/ocr/ocr_space_client.py`

```python
def extract_text_from_image(image_path):
    """
    Send image to OCR.space API and get text back
    
    Returns: {
        'success': bool,
        'raw_text': extracted text,
        'confidence': 0-100% score
    }
    """
```

**File**: `food_guardian_app/ocr/ocr.py`

The `full_extraction()` method orchestrates the OCR pipeline:

```python
def full_extraction(self, image_path):
    """
    Complete OCR extraction pipeline
    
    Steps:
    1. Call OCR.space API
    2. Clean & normalize text
    3. Extract serving size        ← NEW: detects "Serving Size: 30g"
    4. Extract nutrition values    ← Energy, fat, sugars, etc.
    5. Extract ingredients         ← "Sugar, corn syrup, ..."
    6. Extract allergens           ← "Contains: milk, eggs"
    
    Returns: {
        'success': bool,
        'raw_text': original OCR output,
        'cleaned_text': normalized text,
        'serving_size': float (e.g., 30.0),
        'nutrition_values': dict,
        'ingredients': string,
        'allergens': string
    }
    """
```

### Key OCR Features Implemented

#### 1. Serving Size Extraction

**Problem**: Labels can have different serving sizes (30g, 100g, etc.)
**Solution**: Detect and extract serving size before parsing

```python
@staticmethod
def _extract_serving_size(text):
    """
    Looks for patterns like:
    - "Serving Size: 30g"
    - "Per serving (30g)"
    - "One serving = 30g"
    
    Returns: float (e.g., 30.0) or None
    """
    patterns = [
        r'serving\s+size[:\s]+([0-9.]+)\s*g',
        r'per\s+serving\s+\(([0-9.]+)\s*g\)',
        r'one\s+serving\s*=?\s*([0-9.]+)\s*g',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            serving_size = float(match.group(1))
            # Validate: 5g - 500g is reasonable
            if 5 <= serving_size <= 500:
                return serving_size
    return None
```

#### 2. Nutrition Values Extraction

Uses **regex patterns** to find nutrition data in text

```python
@staticmethod
def _extract_nutrition_values(text):
    """
    Uses regex patterns to find:
    - Energy (calories/kcal)
    - Fats (total, saturated, trans)
    - Carbohydrates
    - Sugars
    - Fiber
    - Protein
    - Sodium/Salt
    
    Handles:
    - Multiple formats ("Calories: 110" vs "110 kcal")
    - Decimal separators ("," vs ".")
    - Unit conversions (kJ → kcal)
    - OCR errors and typos
    
    Returns: dict with extracted values
    """
    patterns = {
        'energy_kcal': [
            r'(?:calories?|energy)[:\s]+(\d+(?:\.\d+)?)\s*(?:kcal|cal)',
            r'calories?(?:\s+[a-z]+)*[:\s]+(\d+(?:\.\d+)?)',
        ],
        'fat_g': [
            r'(?:total\s+)?fat[:\s]*(\d+(?:\.\d+)?)\s*g',
            r'fat\s+[•\-\s]*(\d+(?:\.\d+)?)',
        ],
        # ... more patterns for other nutrients
    }
```

#### 3. Text Cleaning

```python
@staticmethod
def _clean_text(text):
    """
    Clean OCR output:
    - Normalize line endings
    - Fix common OCR errors (l→1, O→0)
    - Normalize decimal separators (,→.)
    - Remove extra whitespace
    - Fix spacing around units
    """
```

### OCR Limitations & Robustness

| Factor | Impact | Handling |
|--------|--------|----------|
| Blurry images | High error | User prompted to re-upload |
| Poor lighting | High error | Suggest better image |
| Non-English text | Moderate error | Limited support (English optimized) |
| Handwritten labels | Very high error | Not supported |
| Multiple columns | Moderate error | Regex patterns handle variations |
| OCR errors (typos) | Low error | Pattern matching is flexible |

---

## Data Parsing

### Purpose

After OCR extracts text, we need to **parse** it into structured nutrition data.

### Location

`food_guardian_app/parsing/parser.py`

### The Parsing Process

```
Raw OCR Text (unstructured)
"NUTRITION FACTS
 Serving Size: 30g
 Calories 110
 Total Fat 0g
 Sugars 23g
 ..."
        ↓
[Parse Serving Size]
        ↓
[Extract Nutrition Values]
        ↓
[Scale to Per-100g]  ← NEW: Critical step
        ↓
[Validate Values]
        ↓
Structured Data
{
  'energy_kcal_100g': 366.67,
  'fat_100g': 0.0,
  'sugars_100g': 76.67,
  ...
}
```

### Critical: Per-100g Scaling

**Why?** The model was trained on per-100g values, but labels may show per-serving values.

**Solution**: Scale serving values to per-100g

```python
def _scale_to_per_100g(nutrition_values, serving_size_g):
    """
    If label shows: 30g serving with 23g sugars
    We need to calculate: how much per 100g?
    
    Formula: per_100g = per_serving × (100 / serving_size)
    
    Example:
    - Serving size: 30g
    - Sugars in serving: 23g
    - Scaling factor: 100 / 30 = 3.333
    - Sugars per 100g: 23 × 3.333 = 76.67g
    
    This ensures the model gets consistent per-100g data!
    """
    scaling_factor = 100.0 / serving_size_g
    
    for nutrient in nutrition_values:
        nutrition_values[nutrient] *= scaling_factor
    
    return nutrition_values
```

### Unit Conversions

The parser also handles unit conversions:

```python
# Milligrams → Grams
sodium_g = sodium_mg / 1000

# Kilojoules → Kilocalories
kcal = kj / 4.184

# Sodium → Salt
salt = sodium_g × 2.54
```

### Validation

Each nutrition value is checked:

```python
NUTRIENT_RANGES = {
    'energy_kcal_100g': (0, 900),        # 0-900 kcal reasonable
    'fat_100g': (0, 100),                # Can't exceed 100g per 100g
    'sugars_100g': (0, 100),             # Can't exceed 100g per 100g
    'proteins_100g': (0, 100),           # Can't exceed 100g per 100g
    'salt_100g': (0, 25),                # Reasonable upper limit
}

# If a value is outside range, it's flagged as suspicious
```

### Parse OCR Extraction Method

**File**: `food_guardian_app/parsing/parser.py` (lines ~260-300)

```python
def parse_ocr_extraction(self, ocr_result):
    """
    Main parsing method
    
    Input: OCR result with:
    - raw_text
    - nutrition_values (per serving or per 100g - unknown)
    - serving_size (30g, 100g, etc.)
    - ingredients
    - allergens
    
    Process:
    1. Check if serving_size is provided
    2. If not 100g, scale ALL nutrition values
    3. Standardize field names (energy_kj → energy_kcal_100g)
    4. Validate ranges
    5. Compute nutrient levels (low/moderate/high)
    6. Extract ingredient features
    
    Output: Structured nutrition data ready for model
    """
```

---

## Feature Engineering

### What is Feature Engineering?

Taking raw data and creating **features** (inputs) for the ML model.

**Raw**: Nutrition values, ingredient text
**Features**: 63 numerical inputs for XGBoost

### The 63 Features

#### Group 1: Numeric Features (8 features)

After scaling with StandardScaler:

```python
numeric_cols = [
    'energy_kcal_100g',       # Energy in kcal
    'fat_100g',               # Total fat
    'saturated_fat_100g',     # Saturated fat
    'carbohydrates_100g',     # Total carbs
    'sugars_100g',            # Sugars (important for Nutri-Score)
    'fiber_100g',             # Dietary fiber
    'proteins_100g',          # Protein
    'salt_100g',              # Salt (sodium content)
]
```

**Processing**: StandardScaler normalizes these to mean=0, std=1
- Prevents large values (energy: 400) from dominating small values (salt: 0.5)

#### Group 2: Binary Ingredient Features (5 features)

Extracted from ingredients text:

```python
binary_cols = [
    'additives_count',        # Number of E-numbers (artificial additives)
    'has_sugar',              # Contains sugar/glucose/fructose
    'has_syrup',              # Contains syrup or corn syrup
    'has_palm_oil',           # Contains palm/palmitate oil
    'has_artificial_color',   # Contains artificial coloring
]
```

**How extracted**:

```python
def extract_ingredient_features(ingredients_text):
    """
    Parses ingredient text to extract binary flags
    """
    
    # Count E-numbers (artificial additives)
    e_numbers = re.findall(r'E\d+', ingredients_text.upper())
    additives_count = len(e_numbers)
    
    # Check for keywords
    has_sugar = 1 if any(kw in ingredients.lower() 
                         for kw in ['sugar', 'glucose', 'fructose']) else 0
    
    has_syrup = 1 if 'syrup' in ingredients.lower() else 0
    
    has_palm_oil = 1 if 'palm' in ingredients.lower() else 0
    
    has_artificial_color = 1 if any(color in ingredients.upper() 
                                    for color in ['E110', 'E102', 'TARTRAZINE']) else 0
```

#### Group 3: Text Features (50 features)

TF-IDF Vectorization of ingredients text:

```python
tfidf = TfidfVectorizer(
    max_features=50,           # Keep top 50 terms
    stop_words='english',      # Ignore common words
    min_df=2                   # Ignore terms appearing in <2 documents
)

tfidf_features = tfidf.fit_transform(ingredients_text)
```

**What is TF-IDF?**

TF-IDF measures how important a word is in a document:
- **TF** (Term Frequency): How often does the word appear?
- **IDF** (Inverse Document Frequency): How unique is the word across all documents?

**Example**: If "sugar" appears in 90% of products, it gets low IDF (common). If "xanthan" appears in 2% of products, it gets high IDF (distinctive).

The 50 most important terms from ingredients become features.

### Feature Engineering Code

**File**: `food_guardian_app/parsing/parser.py` (lines ~150-200)

```python
def extract_ingredient_features(self, ingredients_text):
    """Extract binary flags from ingredients"""
    
    sugar_kw = ['sugar', 'sucre', 'azúcar', 'zucchero', 'glucose']
    syrup_kw = ['syrup', 'sirop', 'jarabe', 'fructose']
    palm_kw = ['palm', 'palmitate']
    color_kw = ['e110', 'e102', 'e129', 'tartrazine', 'artificial color']
    
    return {
        'additives_count': self.count_additives(ingredients_text),
        'has_sugar': self.has_ingredient(ingredients_text, sugar_kw),
        'has_syrup': self.has_ingredient(ingredients_text, syrup_kw),
        'has_palm_oil': self.has_ingredient(ingredients_text, palm_kw),
        'has_artificial_color': self.has_ingredient(ingredients_text, color_kw),
    }
```

### Feature Matrix Assembly

```python
# Numeric features (8) - scaled
X_numeric_scaled = scaler.fit_transform(X_numeric)

# Binary features (5) - already 0/1
X_binary = df[binary_cols].values

# Text features (50) - TF-IDF vectors
X_text_vectorized = tfidf.fit_transform(X_text).toarray()

# Combine all
X_combined = np.hstack([X_numeric_scaled, X_binary, X_text_vectorized])
# Result shape: (5172, 63)
```

---

## Model Training

### Training Process

**File**: `COMPLETE_REFACTOR_PIPELINE.py`

#### Step 1: Prepare Data

```
5,898 products → Clean (remove forbidden features) → 5,172 products
                 ↓
            Extract features (63 per product)
                 ↓
            Split 80/20 train/test
                 ↓
       4,137 train samples, 1,035 test samples
```

#### Step 2: Train XGBoost

```python
model = xgb.XGBClassifier(
    objective='multi:softmax',    # Multi-class classification
    num_class=5,                  # 5 output classes (0-4)
    max_depth=6,                  # Tree depth
    learning_rate=0.1,            # Learning rate
    n_estimators=100,             # Number of trees
    random_state=42               # Reproducibility
)

# Train with class weights to handle imbalance
model.fit(X_train, y_train, sample_weight=sample_weights)
```

### Target Variable Mapping

The original dataset uses `nutriscore_letter` with values 1-5:
- 1 = A (Best)
- 2 = B
- 3 = C
- 4 = D
- 5 = E (Worst)

For training, we map to 0-4 (XGBoost convention):

```python
target_mapping = {
    1: 0,  # A
    2: 1,  # B
    3: 2,  # C
    4: 3,  # D
    5: 4   # E
}

reverse_mapping = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E'
}

# After training
y_pred_numeric = model.predict(X_test)  # Returns 0-4
y_pred_letter = reverse_mapping[y_pred_numeric]  # Convert to A-E
```

### Results

```
Training Results:
✓ Accuracy: 83%
✓ 1,035 test samples
✓ 863 correct predictions
✓ 172 incorrect predictions

Class Distribution (Test Set):
A: 19.2%
B: 19.2%
C: 20.3%
D: 19.1%
E: 22.3%  ← Slightly imbalanced (more E products)
```

### Saved Artifacts

After training, 4 files are saved to `model_artifacts/`:

```python
joblib.dump(model, 'best_model_xgboost.pkl')          # The trained model
joblib.dump(scaler, 'feature_scaler.pkl')              # Numeric scaler
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')             # Text vectorizer
json.dump(metadata, 'model_metadata.json')              # Configuration
```

**metadata.json contains**:

```json
{
  "model_accuracy": 0.83,
  "numeric_cols": ["energy_kcal_100g", ...],
  "binary_feature_cols": ["additives_count", ...],
  "tfidf_feature_names": ["sugar", "flour", ...],  // 50 top terms
  "target_mapping": {1: 0, 2: 1, ...},
  "reverse_mapping": {"0": "A", "1": "B", ...},
  "total_features": 63,
  "training_samples": 4137,
  "test_samples": 1035
}
```

---

## Flask Application

### What is Flask?

Flask is a Python web framework. It:
- Handles HTTP requests from users
- Coordinates the ML pipeline
- Serves HTML/CSS/JavaScript
- Returns JSON responses with predictions

### Application Structure

**File**: `food_guardian_app/app.py` (~560 lines)

#### Main Routes

```python
@app.route('/', methods=['GET'])
def index():
    """Serve the main web page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    1. Receive image from user
    2. Run OCR extraction
    3. Parse nutrition data
    4. Return structured data
    """

@app.route('/score', methods=['POST'])
def score():
    """
    1. Receive nutrition data
    2. Load model
    3. Generate prediction
    4. Return Nutri-Score A-E
    """

@app.route('/edit-nutrition', methods=['POST'])
def edit_nutrition():
    """Allow user to manually edit extracted nutrition values"""
```

### Request/Response Flow

#### Request 1: Upload Image

```
User Browser                          Flask App
    │                                   │
    ├─ POST /analyze                   │
    │  (image file)                    │
    │──────────────────────────────────>│
    │                                   ├─ Save image
    │                                   ├─ Run OCR
    │                                   ├─ Parse nutrition
    │                                   │
    │  JSON response                   │
    │<──────────────────────────────────┤
    │  (nutrition data, ingredients)   │
    │                                   │
    └─ Display extracted data           
```

#### Request 2: Score Nutrition

```
User Browser                          Flask App
    │                                   │
    ├─ POST /score                     │
    │  (nutrition values + ingredients)│
    │──────────────────────────────────>│
    │                                   ├─ Load model
    │                                   ├─ Extract features
    │                                   ├─ Run prediction
    │                                   │
    │  JSON response                   │
    │<──────────────────────────────────┤
    │  (Nutri-Score: A-E, confidence)  │
    │                                   │
    └─ Display result with color coding
```

### Key Code: /analyze Endpoint

```python
@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint
    
    Steps:
    1. Handle image upload (file or base64 from camera)
    2. Save image to uploads/ folder
    3. Initialize OCR engine
    4. Run full OCR extraction
    5. Initialize parser
    6. Parse OCR result
    7. Return structured JSON
    """
    
    # Get image (file or base64)
    if 'image' in request.files:
        # File upload
        file = request.files['image']
        image_path = save_file(file)
    elif 'image_base64' in request.form:
        # Camera capture
        base64_str = request.form['image_base64']
        image_path = save_base64_image(base64_str)
    
    # Run OCR
    ocr_engine = get_ocr_engine()
    ocr_result = ocr_engine.full_extraction(image_path)
    
    # Parse
    parser = get_parser()
    parse_result = parser.parse_ocr_extraction(ocr_result)
    
    # Return
    return jsonify({
        'success': True,
        'raw_text': ocr_result['raw_text'],
        'nutrition_data': parse_result['standardized_nutrition'],
        'ingredients': parse_result['ingredients'],
        'serving_size': ocr_result['serving_size'],
        ...
    })
```

### Key Code: /score Endpoint

```python
@app.route('/score', methods=['POST'])
def score():
    """
    Score nutrition data using the model
    
    Input JSON:
    {
        "nutrition_data": {
            "energy_kcal_100g": 366.7,
            "fat_100g": 0.0,
            "sugars_100g": 76.67,
            ...
        },
        "ingredients_text": "SUGAR, CORN SYRUP, ..."
    }
    
    Process:
    1. Load model and artifacts
    2. Call model.predict()
    3. Return A-E score with confidence
    """
    
    data = request.get_json()
    nutrition_data = data['nutrition_data']
    ingredients_text = data['ingredients_text']
    
    # Get model
    model_loader = get_model_loader(model_dir='./model_artifacts')
    
    # Make prediction
    prediction = model_loader.predict(nutrition_data, ingredients_text)
    
    # Return result
    return jsonify({
        'success': True,
        'nutri_score': prediction['nutri_score'],              # 'A', 'B', 'C', 'D', 'E'
        'confidence': prediction['confidence'] * 100,          # 0-100%
        'class_probabilities': prediction['predictions'],      # {A: 0.01, B: 0.01, ...}
        'explanation': explanation_text,
        'recommendation': recommendation_text
    })
```

### Initialization

```python
def initialize_app():
    """Called when Flask app starts"""
    
    # Load model
    model_loader = get_model_loader(model_dir='./model_artifacts')
    print(f"✓ Model loaded, accuracy: {model_loader.metadata['model_accuracy']}")
    
    # Find Tesseract (if Windows)
    if platform.system() == 'Windows':
        tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if os.path.exists(tesseract_path):
            print(f"✓ Found Tesseract at {tesseract_path}")
    
    # Initialize OCR engine
    ocr_engine = get_ocr_engine(tesseract_path=tesseract_path)
    print("✓ OCR Engine initialized")
    
    # Initialize parser
    parser = get_parser()
    print("✓ Parser initialized")

if __name__ == '__main__':
    initialize_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
```

---

## Model Loading & Prediction

### The Refactored Model Loader

**File**: `food_guardian_app/model/refactored_model_loader.py`

This is the **most important file** for production deployment.

### Purpose

1. **Load pre-trained artifacts** (model, scaler, vectorizer)
2. **Validate inputs** (reject forbidden features)
3. **Feature engineering** (scale, vectorize)
4. **Make predictions** (run through XGBoost)
5. **Return results** (A-E score with confidence)

### Key Method: predict()

```python
def predict(self, nutrition_data, ingredients_text):
    """
    Make Nutri-Score prediction for a product
    
    Input:
    - nutrition_data: dict with 8 nutrition values
        {
            'energy_kcal_100g': 366.7,
            'fat_100g': 0.0,
            'saturated_fat_100g': 0.0,
            'carbohydrates_100g': 90.0,
            'sugars_100g': 76.67,
            'fiber_100g': 0.0,
            'proteins_100g': 0.0,
            'salt_100g': 0.21
        }
    - ingredients_text: string with ingredients
    
    Process:
    1. Validate no forbidden features
    2. Preprocess nutrition (clipping to ranges)
    3. Extract ingredient features (5 binary)
    4. Scale numeric features
    5. Vectorize ingredients text (50 TF-IDF)
    6. Combine all 63 features
    7. Run through model
    8. Get probabilities for all 5 classes
    9. Convert to letter (0→A, 1→B, etc.)
    10. Return with confidence
    
    Output:
    {
        'nutri_score': 'E',
        'confidence': 0.78,
        'predictions': {'A': 0.001, 'B': 0.003, 'C': 0.010, 'D': 0.204, 'E': 0.781}
    }
    """
```

### Input Validation

```python
def validate_input_features(self, input_data):
    """
    Reject any forbidden features
    
    FORBIDDEN_FEATURES = {
        'image_path', 'brand', 'country', 'ecoscore_grade',
        'nova_group', 'additives', 'product_id', 'barcode',
        # ... 30+ more
    }
    
    If user tries to pass forbidden feature:
    ❌ ValueError: "FORBIDDEN FEATURES DETECTED: {image_path, brand}"
    """
    input_keys = set(input_data.keys())
    forbidden_found = input_keys & FORBIDDEN_FEATURES
    
    if forbidden_found:
        raise ValueError(f"❌ FORBIDDEN FEATURES: {forbidden_found}")
```

### Feature Engineering in Prediction

```python
def predict(self, nutrition_data, ingredients_text):
    # 1. Validate
    self.validate_input_features(nutrition_data)
    
    # 2. Preprocess nutrition (clip to reasonable ranges)
    nutrition_data = self.preprocess_nutrition_values(nutrition_data)
    
    # 3. Extract ingredient features (5 features)
    ingredient_features = self.extract_ingredient_features(ingredients_text)
    
    # 4. Build numeric features (8 features)
    numeric_values = [nutrition_data.get(col, 0) for col in numeric_cols]
    numeric_array = np.array(numeric_values).reshape(1, -1)
    
    # 5. Scale numeric features
    numeric_scaled = self.scaler.transform(numeric_array)[0]
    
    # 6. Build binary features (5 features)
    binary_values = [ingredient_features[col] for col in binary_cols]
    
    # 7. Get TF-IDF features (50 features)
    tfidf_sparse = self.tfidf_vectorizer.transform([ingredients_text])
    tfidf_values = tfidf_sparse.toarray()[0]
    
    # 8. Combine all (8 + 5 + 50 = 63 features)
    all_features = np.concatenate([numeric_scaled, binary_values, tfidf_values])
    all_features = all_features.reshape(1, -1)
    
    # 9. Make prediction
    pred_numeric = self.xgb_model.predict(all_features)[0]        # Returns 0-4
    pred_proba = self.xgb_model.predict_proba(all_features)[0]    # Returns [0.001, 0.003, ...]
    
    # 10. Convert to letter
    reverse_map = self.metadata['reverse_mapping']
    pred_letter = reverse_map[str(int(pred_numeric))]              # 0→'A', 1→'B', etc.
    
    # 11. Get confidence (max probability)
    confidence = float(np.max(pred_proba))
    
    # 12. Return result
    return {
        'nutri_score': pred_letter,
        'confidence': confidence,
        'predictions': {
            reverse_map[str(i)]: float(prob)
            for i, prob in enumerate(pred_proba)
        }
    }
```

### How Artifacts are Loaded

```python
def load_all_artifacts(self):
    """
    Load pre-trained model components
    
    These files were created during training (COMPLETE_REFACTOR_PIPELINE.py)
    """
    
    # 1. Load trained XGBoost model
    self.xgb_model = joblib.load('model_artifacts/best_model_xgboost.pkl')
    # This is a fitted XGBClassifier object
    # It has learned: feature_weights, tree_structures, etc.
    
    # 2. Load feature scaler
    self.scaler = joblib.load('model_artifacts/feature_scaler.pkl')
    # StandardScaler fitted on training data
    # Remember: mean and std for each of 8 numeric features
    
    # 3. Load TF-IDF vectorizer
    self.tfidf_vectorizer = joblib.load('model_artifacts/tfidf_vectorizer.pkl')
    # TfidfVectorizer fitted on training ingredients text
    # Remember: which terms are important, their weights
    
    # 4. Load metadata
    with open('model_artifacts/model_metadata.json', 'r') as f:
        self.metadata = json.load(f)
    # Contains: feature names, target mappings, model accuracy, etc.
```

### Singleton Pattern

```python
# Global instance
_model_loader = None

def get_model_loader(model_dir='./model_artifacts'):
    """
    Get or create model loader instance (singleton)
    
    Why singleton?
    - Loading models is slow (100ms+)
    - Reuse same instance across multiple predictions
    - Save time by not reloading on every request
    
    First call: Load from disk (100ms)
    Subsequent calls: Return cached instance (1ns)
    """
    global _model_loader
    if _model_loader is None:
        _model_loader = RefactoredModelLoader(model_dir)
    return _model_loader
```

---

## End-to-End Workflow

### Complete User Journey

```
1. USER UPLOADS NUTRITION LABEL IMAGE
   ↓
   [Image saved to uploads/ folder]
   ↓
2. FLASK RECEIVES /analyze REQUEST
   ↓
3. OCR EXTRACTION (ocr.py)
   └─ _extract_serving_size()     → "30g"
   └─ _extract_nutrition_values() → {energy: 110, sugars: 23, ...}
   └─ _extract_ingredients()      → "SUGAR, CORN SYRUP, ..."
   └─ _extract_allergens()        → "CONTAINS: MILK"
   ↓
   Returns: OCR result with serving_size, nutrition_values, ingredients
   ↓
4. DATA PARSING (parser.py)
   └─ serving_size = 30g detected
   └─ scaling_factor = 100 / 30 = 3.333
   └─ Scale all nutrition values × 3.333
   └─ Per-100g: {energy: 366.7, sugars: 76.67, ...}
   ↓
   Returns: Parsed nutrition data (per 100g)
   ↓
5. FLASK RETURNS /analyze RESPONSE
   ├─ Nutrition table (energy, fats, sugars, etc.)
   ├─ Ingredients list
   ├─ Serving size detected
   └─ Allow user to edit if needed
   ↓
6. USER REVIEWS & CLICKS "SCORE NUTRITION"
   ↓
7. FLASK RECEIVES /score REQUEST
   ├─ nutrition_data: {energy: 366.7, fat: 0.0, sugars: 76.67, ...}
   └─ ingredients_text: "SUGAR, CORN SYRUP, ..."
   ↓
8. MODEL LOADING (refactored_model_loader.py)
   └─ Load best_model_xgboost.pkl     (trained XGBoost)
   └─ Load feature_scaler.pkl          (StandardScaler)
   └─ Load tfidf_vectorizer.pkl        (TF-IDF)
   └─ Load model_metadata.json         (config)
   ↓
9. FEATURE ENGINEERING
   ├─ Preprocess nutrition values (clip to ranges)
   ├─ Scale numeric features [8] → StandardScaler
   ├─ Extract ingredient features [5]
   │  ├─ count_additives() → E-number count
   │  ├─ has_sugar() → 1 or 0
   │  ├─ has_syrup() → 1 or 0
   │  ├─ has_palm_oil() → 1 or 0
   │  └─ has_artificial_color() → 1 or 0
   ├─ Vectorize ingredients [50] → TF-IDF
   └─ Combine: [8 + 5 + 50 = 63 features]
   ↓
10. MODEL PREDICTION
    └─ XGBoost.predict(63 features)
       ├─ Input: [366.7, 0.0, 0.0, 90.0, 76.67, 0.0, 0.0, 0.21, 1, 1, 0, 1, 1, 0.01, ...]
       ├─ Process: 100+ decision trees voting
       └─ Output: Class 4 (internal representation)
    └─ XGBoost.predict_proba(63 features)
       └─ Output: [0.001, 0.003, 0.010, 0.204, 0.781]
                   (probabilities for A, B, C, D, E)
    ↓
11. CONVERT TO NUTRI-SCORE
    └─ reverse_mapping[4] = 'E'
    └─ confidence = max(probabilities) = 0.781 = 78.1%
    ↓
12. FLASK RETURNS /score RESPONSE
    ├─ Nutri-Score: E
    ├─ Confidence: 78.1%
    ├─ Breakdown: {A: 0.1%, B: 0.3%, C: 1.0%, D: 20.4%, E: 78.1%}
    ├─ Color: Red (E is worst)
    └─ Recommendation: "Very high sugar content. Avoid."
    ↓
13. FRONTEND DISPLAYS RESULTS
    ├─ Large "E" in red
    ├─ Confidence bar
    ├─ Class probability visualization
    └─ Health recommendation
    ↓
14. USER SEES FINAL RESULT! ✓
```

### Example: Candy Product

```
REAL WORLD EXAMPLE
==================

1. User uploads candy nutrition label
   (shows per 30g serving)

2. OCR extraction:
   - Serving Size: 30g ✓
   - Calories: 110
   - Fat: 0g
   - Carbs: 27g
   - Sugars: 23g
   - Protein: 0g

3. Parser scales to per-100g:
   - Calories: 110 × (100/30) = 366.7 ✓
   - Fat: 0 × 3.333 = 0.0 ✓
   - Carbs: 27 × 3.333 = 90.0 ✓
   - Sugars: 23 × 3.333 = 76.67 ✓
   - Protein: 0 × 3.333 = 0.0 ✓

4. Feature engineering:
   - Numeric: [366.7, 0.0, 0.0, 90.0, 76.67, 0.0, 0.0, 0.083]
   - Binary: [1 (additives), 1 (has_sugar), 1 (has_syrup), 0 (palm), 1 (color)]
   - Text: [0.2, 0.0, 0.1, ..., 0.05] (50 TF-IDF features)
   - Total: 63 features ✓

5. Model prediction:
   - Probabilities: [0.1%, 0.3%, 1.0%, 20.4%, 78.1%]
   - Highest: 78.1% for class E
   - Result: Nutri-Score E (worst) ✓

6. User sees:
   "Nutri-Score: E (Very High Sugar)
    This product contains extremely high amounts of sugar.
    Not recommended for regular consumption."

7. CORRECT! Sugar-only candy gets E ✓
```

---

## Key Takeaways for Your Discussion

### 1. **The Core Problem Solved**
- Original model used forbidden features (images, brands, ecoscore)
- Refactored to use **ONLY nutrition label data**
- Now works in real-world scenarios

### 2. **The Critical Insight**
- Labels show **per-serving values** (e.g., 30g)
- Model trained on **per-100g values**
- Solution: **Detect serving size and scale automatically**
- This was the bug fix you implemented!

### 3. **OCR Pipeline**
- Uses **OCR.space API** (cloud-based, reliable)
- Falls back to **Tesseract-OCR** (local, optional)
- Extracts: nutrition values, serving size, ingredients, allergens

### 4. **Feature Engineering**
- **8 numeric** nutrition features (scaled)
- **5 binary** ingredient flags (additives, sugar, syrup, palm, color)
- **50 text** features (TF-IDF of ingredient words)
- Total: **63 features** fed to XGBoost

### 5. **Model Architecture**
- **XGBoost classifier** (100 decision trees)
- **5 output classes** (A, B, C, D, E)
- **83% accuracy** on test set
- Learns: which combinations of nutrients → which Nutri-Score

### 6. **Flask App Coordination**
- `/analyze` endpoint: Image → OCR → Parse → Return nutrition
- `/score` endpoint: Nutrition → Model → Return A-E score
- Singleton pattern: Load model once, reuse for all requests

### 7. **Production Ready**
- Validates forbidden features
- Handles missing values
- Clips unreasonable values
- Provides confidence scores
- Beautiful web interface

---

## Questions You Might Get Asked

### Q: "Why not use images?"
A: Because the model should work with text-only nutrition labels. Images introduce bias (packaging design, brand colors) unrelated to actual nutrition.

### Q: "Why 83% accuracy?"
A: That's typical for multi-class classification with real-world data. The model sees subtle nutritional patterns that distinguish products.

### Q: "What if OCR fails?"
A: User can manually edit extracted values. The app is semi-manual - OCR is a starting point, not gospel.

### Q: "Why scale to per-100g?"
A: Standardization. Labels show different serving sizes (30g, 100g, 1 cup, etc.). Converting to per-100g ensures consistent model input.

### Q: "What happens with partial data?"
A: The parser fills missing values with 0 (conservative). The model still makes a prediction, but might be less confident.

### Q: "How fast are predictions?"
A: ~100ms after model loads. First request: 200ms (includes load). Subsequent requests: 100ms (reuses loaded model).

---

## Architecture Summary Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                  Smart Food Guardian                         │
│                   (Complete System)                          │
└──────────────────────────────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                ↓                         ↓
            USER INTERFACE         BACKEND SYSTEM
         (HTML/CSS/JS)            (Python/Flask)
              │                        │
              │                ┌───────┴────────┐
              │                ↓                ↓
              │            OCR MODULE       PARSER MODULE
              │            (ocr.py)        (parser.py)
              │                │                │
              │        ┌───────┴────────┐      │
              │        ↓                ↓      │
              │   OCR.space        Tesseract   │
              │   (Cloud)          (Local)     │
              │        │                │      │
              │        └────────┬───────┘      │
              │                 ↓              │
              │            Extract:            │
              │         - Serving size        │
              │         - Nutrition           │
              │         - Ingredients         │
              │         - Allergens           │
              │                 │              ↓
              │                 │    ┌─────────────────┐
              │                 │    │ SCALE to Per-100g
              │                 └───→│ VALIDATE values
              │                      │ EXTRACT features
              │                      └────────┬────────┘
              │                               ↓
              │                    MODEL LOADING & FEATURES
              │                    (refactored_model_loader.py)
              │                               │
              │                    ┌──────────┴──────────┐
              │                    ↓                     ↓
              │            Load Artifacts         Feature Engineering
              │                    │                     │
              │         ┌──────────┼──────────┐         │
              │         ↓          ↓          ↓         │
              │      Model      Scaler     TF-IDF       │
              │      (pkl)      (pkl)      (pkl)        │
              │         │          │         │          │
              │         └──────────┼─────────┘          │
              │                    │                    │
              │                    └────────┬───────────┘
              │                             ↓
              │                    ┌────────────────────┐
              │                    │  63 Features:      │
              │                    │  - 8 Numeric       │
              │                    │  - 5 Binary        │
              │                    │  - 50 TF-IDF       │
              │                    └──────────┬─────────┘
              │                               ↓
              │                        XGBoost Model
              │                     (Trained Classifier)
              │                               ↓
              │                        Prediction
              │                      (Class 0-4)
              │                               ↓
              │                    ┌─────────────────────┐
              │                    │  Reverse Map:       │
              │                    │  0 → 'A' (Best)     │
              │                    │  1 → 'B'            │
              │                    │  2 → 'C'            │
              │                    │  3 → 'D'            │
              │                    │  4 → 'E' (Worst)    │
              │                    └──────────┬──────────┘
              │                               ↓
              │←──────────────────  Nutri-Score Result
              │                   + Confidence %
              │                   + Recommendations
              ↓
         Display to User
         (Color-coded card)
         (Probability bars)
         (Health advice)
```

Good luck with your discussion! You now have all the details to explain the entire system. 🍎
