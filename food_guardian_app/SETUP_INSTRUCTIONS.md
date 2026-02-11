# Smart Food Guardian - Setup Instructions

## System Requirements

- **Python 3.8+** (3.10+ recommended)
- **Tesseract-OCR** (optional - uses OCR.space API as fallback)
- **Windows/Mac/Linux** compatible

## Installation Steps

### 1. Clone or Extract the Project

```bash
cd d:\zzz\food_guardian_app
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

**Activate it:**
- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install Tesseract-OCR

**Windows:**
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run installer (keep default installation path: `C:\Program Files\Tesseract-OCR`)
3. The app will auto-detect it

**Mac:**
```bash
brew install tesseract
```

**Linux (Ubuntu):**
```bash
sudo apt-get install tesseract-ocr
```

**Note:** OCR.space API is used as primary fallback, so Tesseract is optional.

### 5. Verify Installation

```bash
python -c "from ocr.ocr import get_ocr_engine; print('✓ OCR OK')"
python -c "from model.refactored_model_loader import get_model_loader; print('✓ Model OK')"
```

## Running the Application

### Start Flask Server

```bash
python app.py
```

Expected output:
```
✓ Refactored model loader initialized
✓ OCR Engine initialized  
✓ Parser initialized
Running on http://localhost:5000
```

### Access the Web App

Open your browser and go to: **http://localhost:5000**

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'flask'"

**Solution:** Install dependencies again
```bash
pip install -r requirements.txt
```

### Issue: "Tesseract is not installed or not in PATH"

**Solution:** 
- The app uses OCR.space API automatically, so no action needed
- Or install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki

### Issue: "Model files not found"

**Solution:** Ensure these files exist in `model_artifacts/`:
- `best_model_xgboost.pkl`
- `feature_scaler.pkl`
- `tfidf_vectorizer.pkl`
- `model_metadata.json`

If missing, run the training script (see ROOT README.md)

### Issue: Port 5000 already in use

**Solution:** Change port in `app.py` line 543:
```python
app.run(port=5001)  # Use different port
```

## Testing the Model

### Quick Test
```bash
python test_nutrients.py
```

### OCR Test
```bash
python test_ocr_extraction.py
```

## Configuration

Edit `config.ini` to customize:
- Upload folder location
- Allowed file types
- Model parameters
- OCR settings

## Next Steps

1. Upload a nutrition label image
2. Review extracted data
3. Edit if needed
4. Click "Score Nutrition"
5. View Nutri-Score prediction!

## Support

See:
- `DELIVERY_SUMMARY.txt` - Project overview
- `CHECKLIST.txt` - Feature status
- `diagnose.py` - Diagnostic tools
- ROOT `README.md` - Full documentation

**Questions?** Check the test files for usage examples.
