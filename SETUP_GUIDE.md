# 🚀 Complete Setup Guide - AI Voice Detection API

## 📋 Prerequisites

- **Python:** 3.8 or higher
- **System RAM:** 4GB minimum (8GB recommended)
- **Disk Space:** 2GB free (for ML models)
- **Internet:** Required for first-time model download

---

## ✅ Step 1: Clone/Navigate to Project

```powershell
# Navigate to your project directory
cd "C:\Users\nisha\Documents\1.CODING\HCL HACKTHONE\ai-audio-detector-main\ai-audio-detector-main"
```

---

## ✅ Step 2: Create Virtual Environment

### Option A: Using venv (Recommended)

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (PowerShell)
venv\Scripts\Activate.ps1

# Your prompt should now show (venv) at the start
```

### Option B: Using Conda

```powershell
# Create conda environment
conda create -n ai-detector python=3.9

# Activate conda environment
conda activate ai-detector
```

---

## ✅ Step 3: Verify Python Installation

```powershell
# Check Python version
python --version

# Should show: Python 3.x.x
```

---

## ✅ Step 4: Install Dependencies

```powershell
# Make sure you're in the virtual environment first!
# (You should see (venv) or (ai-detector) in your prompt)

# Install from requirements.txt
pip install -r requirements.txt
```

**If you get errors:**
```powershell
# Try with prefer-binary flag
pip install -r requirements.txt --prefer-binary

# Or install packages individually
pip install fastapi uvicorn requests librosa numpy scipy
pip install torch torchaudio transformers
```

---

## ✅ Step 5: Verify Installation

```powershell
# Check if FastAPI installed
python -c "import fastapi; print(fastapi.__version__)"

# Check if torch installed
python -c "import torch; print(torch.__version__)"

# Check if transformers installed
python -c "import transformers; print(transformers.__version__)"
```

---

## ✅ Step 6: Start the Server

```powershell
# Make sure you're in the virtual environment
# (You should see (venv) in your prompt)

# Navigate to backend folder
cd backend

# Run the server
python main.py
```

**Expected Output:**
```
============================================================
AI Voice Detection API
============================================================

Starting server...
API Documentation: http://localhost:8000/docs
OpenAPI Schema: http://localhost:8000/openapi.json

Supported Languages: Tamil, English, Hindi, Malayalam, Telugu
Authentication: x-api-key header (ai-audio-detection-akatsukiapi)

Press CTRL+C to stop the server

INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## ✅ Step 7: Test the API (3 Methods)

### Method 1: Web Browser (Interactive Docs) ⭐ EASIEST

1. Open browser: http://localhost:8000/docs
2. Click on **POST /api/voice-detection**
3. Click **Try it out**
4. Fill in:
   - **x-api-key:** `ai-audio-detection-akatsukiapi`
   - **Request body:** (see examples below)
5. Click **Execute**

### Method 2: Command Line (Test Client)

First, prepare your MP3 file:
- Place `sample.mp3` in the project root folder

Then run:
```powershell
# From project root (NOT backend folder)
# Close the server first (Ctrl+C), or open new terminal

# Default API key
python test_voice_detection.py sample.mp3 English

# Custom API key
python test_voice_detection.py sample.mp3 Tamil ai-audio-detection-akatsukiapi
```

### Method 3: Python Script

Create a file `test_api.py`:

```python
import requests
import base64
import json

# Read MP3 file and convert to Base64
with open("sample.mp3", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode("utf-8")

# Prepare request
url = "http://localhost:8000/api/voice-detection"
headers = {
    "Content-Type": "application/json",
    "x-api-key": "ai-audio-detection-akatsukiapi"
}

payload = {
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": audio_base64
}

# Send request
response = requests.post(url, json=payload, headers=headers)
result = response.json()

# Print response
print(json.dumps(result, indent=2))
```

Run it:
```powershell
python test_api.py
```

---

## 📝 Request Format

```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
}
```

**Supported Languages:**
- `Tamil`
- `English`
- `Hindi`
- `Malayalam`
- `Telugu`

---

## 📤 Response Format

### Success Response (HTTP 200)

```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.91,
  "explanation": "Unnatural pitch consistency and robotic speech patterns detected"
}
```

### Error Response (HTTP 4xx/5xx)

```json
{
  "status": "error",
  "message": "Invalid API key or malformed request"
}
```

---

## 🔑 API Key

**Default API Key:** `ai-audio-detection-akatsukiapi`

**Where to use it:**
- In HTTP header: `x-api-key: ai-audio-detection-akatsukiapi`

**Change API Key:**
Edit `backend/main.py` and modify:
```python
VALID_API_KEYS = {
    "your-new-api-key",
    "another-key"
}
```

---

## 📁 Project Structure

```
ai-audio-detector-main/
│
├── backend/
│   ├── main.py              # FastAPI application
│   ├── models.py            # ML model loading & inference
│   ├── utils.py             # Audio preprocessing utilities
│   └── __init__.py
│
├── venv/                    # Virtual environment (auto-created)
│
├── test_voice_detection.py  # Command-line test tool
├── requirements.txt         # Python dependencies
├── README.md               # Project overview
├── API_DOCUMENTATION.md    # Full API docs
├── SETUP_GUIDE.md         # This file
└── HACKATHON_README.md    # Hackathon instructions
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'fastapi'"

**Solution:**
```powershell
# Make sure virtual environment is activated
# You should see (venv) in your prompt

# Reinstall requirements
pip install -r requirements.txt
```

### Issue: "API not responding at http://localhost:8000"

**Solution:**
```powershell
# Check if server is running
# You should see output in terminal

# If not, start it again
cd backend
python main.py
```

### Issue: "Models not loading - takes too long"

**Solution:**
- First run downloads ~500MB of models (one-time only)
- This can take 5-10 minutes on slow internet
- Subsequent runs will be fast

### Issue: "Port 8000 already in use"

**Solution:**
```powershell
# Kill the process using port 8000
# Or change port in backend/main.py

# Edit last line:
# uvicorn.run(..., port=8001)  # Use 8001 instead
```

### Issue: "Invalid API key error"

**Solution:**
- Check spelling: `ai-audio-detection-akatsukiapi` (exact case)
- Verify it's in the `x-api-key` header, not request body

---

## 🚀 Quick Start (TL;DR)

```powershell
# 1. Activate virtual environment
venv\Scripts\Activate.ps1

# 2. Install dependencies (if not already done)
pip install -r requirements.txt

# 3. Start server
cd backend
python main.py

# 4. In another terminal, test API
# Option A: Web browser
# Open http://localhost:8000/docs

# Option B: Command line
# python test_voice_detection.py sample.mp3 English

# Option C: Python
# python test_api.py
```

---

## 📚 API Examples

### Example 1: Detect AI-Generated Tamil Voice

**Request:**
```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2..."
}
```

**Response:**
```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.89,
  "explanation": "Consistent pitch patterns and absence of natural speech variations detected. Phoneme transitions appear mechanical."
}
```

### Example 2: Detect Human Hindi Voice

**Request:**
```json
{
  "language": "hindi",
  "audioFormat": "mp3",
  "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2..."
}
```

**Response:**
```json
{
  "status": "success",
  "language": "Hindi",
  "classification": "HUMAN",
  "confidenceScore": 0.92,
  "explanation": "Natural speech variations and authentic breathing patterns detected. Emotional expression and prosody indicate human speaker."
}
```

---

## ✅ Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Virtual environment activated
- [ ] `pip install -r requirements.txt` completed
- [ ] Server running: `python main.py`
- [ ] API accessible: http://localhost:8000/docs
- [ ] API key noted: `ai-audio-detection-akatsukiapi`
- [ ] Test request successful
- [ ] Response contains classification and confidence score

---

## 📞 Need Help?

1. **Check API docs:** http://localhost:8000/docs
2. **Read full documentation:** [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
3. **Check logs:** Look at terminal output where server is running
4. **Verify audio file:** MP3 must be valid and not corrupted

---

## 🎉 Done!

Your API is now ready for:
- ✅ Testing
- ✅ Integration
- ✅ Hackathon submission
- ✅ Production deployment

**Start by testing at:** http://localhost:8000/docs

Good luck! 🚀
