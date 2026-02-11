# Smart Food Guardian - Setup & Usage Guide

A production-ready web application for analyzing nutrition labels using OCR and XGBoost machine learning.

## 📋 Project Structure

```
food_guardian_app/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── model/
│   └── model_loader.py    # Model loading & inference
├── ocr/
│   └── ocr.py             # OCR & text extraction
├── parsing/
│   └── parser.py          # Nutrition data parsing
├── templates/
│   └── index.html         # Web interface
├── static/
│   └── style.css          # Styling
├── uploads/               # Uploaded images (auto-created)
└── model_artifacts/       # Trained model files (auto-created)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd food_guardian_app
pip install -r requirements.txt
```

**Note for Windows Users**: You also need to install Tesseract OCR:
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Install to default location: `C:\Program Files\Tesseract-OCR`
- The app will auto-detect it

**Note for Mac/Linux Users**: Install Tesseract via:
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr
```

### 2. Train & Save the Model

Run the Jupyter notebook `best_model_xgboost.ipynb` to train the model. At the **end** of the notebook, add this code:

```python
# Save model artifacts for web app
from model.model_loader import get_model_loader

model_loader = get_model_loader(model_dir='./model_artifacts')
model_loader.save_model_artifacts(
    xgb_model=xgb,           # The trained XGBoost model
    tokenizer=tokenizer,      # Keras tokenizer
    scaler=scaler,            # StandardScaler
    text_extractor=text_extractor,  # Text embedding model
    tab_extractor=tab_extractor     # Tabular embedding model
)
```

This creates a `model_artifacts/` folder with:
- `xgb_model.pkl` - Trained XGBoost model
- `tokenizer.pkl` - Text tokenizer
- `scaler.pkl` - Feature scaler
- `text_extractor.h5` - Text embedding neural network
- `tab_extractor.h5` - Tabular embedding neural network

**Copy this folder to** `food_guardian_app/model_artifacts/`

### 3. Run the Web App

```bash
cd food_guardian_app
python app.py
```

Visit: **http://localhost:5000**

## 🎯 How to Use

### Capture from Camera
1. Click **"📷 Camera"** tab
2. Click **"Start Camera"** button
3. Position nutrition label in view
4. Click **"📸 Capture Photo"**
5. Click **"🔍 Analyze Image"**

### Upload Image
1. Click **"📁 Upload File"** tab
2. Drag & drop or click to select image
3. Click **"🔍 Analyze Image"**

### Review & Edit Results
1. **View extracted text** - See what OCR detected
2. **Edit nutrition values** - Click ✏️ to modify extracted numbers
3. **Review ingredients** - See detected ingredients and allergens

### Get Nutrition Score
1. Click **"⭐ Get Nutri-Score"**
2. See results:
   - **Nutri-Score** (A-E scale)
   - **Confidence** percentage
   - **Score distribution** chart
   - **Recommendation** based on score

## 📝 API Endpoints

### GET `/`
Returns HTML home page with camera/upload interface.

### POST `/analyze`
Analyzes a nutrition label image.

**Request:**
```
Form data:
- image: Image file (or)
- image_base64: Base64 encoded image from camera
```

**Response:**
```json
{
  "success": true,
  "raw_text": "extracted text",
  "cleaned_text": "cleaned text",
  "extracted_data": {
    "ingredients": "...",
    "allergens": "...",
    "nutrition_values": {
      "Energy": "150 kcal",
      "Total Fat": "10 g",
      ...
    }
  },
  "nutrition_data": { raw nutrition dict },
  "nutrient_levels": { nutrient levels },
  "validation": { validation messages }
}
```

### POST `/score`
Scores nutrition data using the trained model.

**Request:**
```json
{
  "nutrition_data": {
    "energy_kcal_100g": 150,
    "fat_100g": 10,
    "saturated_fat_100g": 5,
    ...
  },
  "ingredients": "list of ingredients",
  "allergens": "allergen info"
}
```

**Response:**
```json
{
  "success": true,
  "nutri_score": "A",
  "confidence": 92.5,
  "class_probabilities": {
    "A": 92.5,
    "B": 5.0,
    "C": 2.0,
    "D": 0.5,
    "E": 0.0
  },
  "explanation": {
    "color": "#2ecc71",
    "title": "Excellent Nutrition",
    "description": "...",
    "recommendation": "..."
  }
}
```

### POST `/edit-nutrition`
Allows user to edit extracted nutrition values.

### GET `/health`
Health check endpoint.

## 🔧 Configuration

### Tesseract Path (Windows)
If installed to non-standard location, update in `app.py`:

```python
ocr_engine = get_ocr_engine(tesseract_path=r'C:\Path\To\Tesseract-OCR\tesseract.exe')
```

### Upload Settings
In `app.py`:
```python
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB
```

### Model Location
By default, looks for `./model_artifacts/` in app directory.

## 🐛 Troubleshooting

### "Model not available" Error
- **Fix**: Make sure you've run the training notebook and saved model artifacts
- Copy `model_artifacts/` folder to `food_guardian_app/`

### "Could not access camera" Error
- Browser needs permission to access camera
- Check browser settings for https://localhost:5000
- Use HTTPS or localhost only

### Tesseract Not Found
**Windows:**
- Install from: https://github.com/UB-Mannheim/tesseract/wiki
- Use default install path

**Mac:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

### OCR Results are Poor
- **Improve image quality**: Better lighting, focus on label
- **Positioning**: Keep label straight and fill frame
- **Resolution**: Higher resolution images work better
- **Contrast**: High contrast between text and background

### Model Predictions Seem Off
- Verify training notebook completed successfully
- Check that all required features are provided in `/score` request
- Ensure nutrition values are in correct ranges

## 📊 Model Architecture

The model uses a **late fusion** approach:

1. **Text Component**
   - Tokenized ingredient/allergen text
   - 64-dim embedding from Keras LSTM/Dense layers
   - Extracted from hidden layer

2. **Tabular Component**
   - 24 nutritional & eco features
   - StandardScaler normalized
   - 256-dim embedding from Keras Dense layers
   - Extracted from hidden layer

3. **Fusion**
   - Concatenate embeddings (64 + 256 = 320 features)
   - Feed to XGBoost classifier

4. **Output**
   - Multi-class classification (A, B, C, D, E)
   - Probability distribution over 5 classes

## 📁 File Uploads

Uploaded images are saved with timestamp:
```
uploads/
├── 20240123_153045_nutrition_label.jpg
├── 20240123_153102_camera_capture.jpg
└── ...
```

To clean up old uploads:
```bash
# Linux/Mac
find uploads/ -mtime +7 -delete

# Windows
powershell: Get-ChildItem uploads/ -Include *.jpg,*.png -Recurse | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-7)} | Remove-Item
```

## 🌐 Deployment

### Local Network Access
To allow other devices on your network to access:

```bash
python app.py  # Already listens on 0.0.0.0:5000
```

Access from another device:
```
http://<your-computer-ip>:5000
```

Find your IP:
```bash
# Windows
ipconfig

# Mac/Linux
ifconfig
```

### Production Deployment
For production, use a proper WSGI server:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Or with Flask built-in (development only):
```bash
python app.py
```

## 📖 Module Documentation

### `model/model_loader.py`
Handles trained model loading and inference.

**Key Classes:**
- `ModelLoader` - Loads and manages model artifacts
- Functions: `predict()`, `batch_predict()`, `extract_embeddings()`

### `ocr/ocr.py`
Extracts text from nutrition label images using Tesseract.

**Key Classes:**
- `NutritionOCR` - Main OCR engine
- Functions: `extract_from_image()`, `extract_nutrition_values()`, `extract_ingredients()`

### `parsing/parser.py`
Parses and structures nutrition data.

**Key Classes:**
- `NutritionParser` - Nutrition data parser
- Functions: `parse_ocr_extraction()`, `compute_derived_features()`, `validate_nutrition_values()`

## 🎨 Frontend Features

✅ **Responsive Design** - Works on desktop, tablet, mobile
✅ **Real-time Camera** - Capture directly from device
✅ **Drag & Drop Upload** - Intuitive file handling
✅ **Live Editing** - Edit nutrition values before scoring
✅ **Beautiful UI** - Modern gradient design with icons
✅ **Progress Feedback** - Loading spinners and validation messages
✅ **Score Visualization** - Probability charts and recommendations

## 📄 License

This project is part of Smart Food Guardian.

## 🤝 Support

For issues or questions:
1. Check troubleshooting section above
2. Review logs in Flask console
3. Ensure all dependencies are installed
4. Verify model artifacts are in correct location

---

**Happy analyzing! 🍎**
