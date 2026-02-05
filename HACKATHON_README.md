# 🎙️ AI Voice Detection API - Hackathon Project

## 🎯 Project Overview

This is a production-ready REST API that detects whether voice audio is **AI-Generated** or **Human-Spoken** across 5 supported languages:
- Tamil
- English
- Hindi
- Malayalam
- Telugu

**Features:**
✅ Base64 MP3 input support  
✅ API Key authentication  
✅ Accurate AI vs Human classification  
✅ High confidence scores (0.0-1.0)  
✅ Detailed explanations for each classification  
✅ Fast processing (<30s per audio)  
✅ Production-ready error handling

## 📋 Requirements

- **Python:** 3.8 or higher
- **RAM:** 4GB minimum (8GB recommended)
- **Disk Space:** 2GB for models
- **Internet:** Required for first-time model download only

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
cd backend
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
Authentication: x-api-key header (sk_test_123456789)

Press CTRL+C to stop the server
```

### 3. Test the API

**Option A: Using Test Client**
```bash
python test_voice_detection.py sample.mp3 Tamil sk_test_123456789
```

**Option B: Using cURL**
```bash
curl -X POST http://localhost:8000/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_test_123456789" \
  -d '{
    "language": "Tamil",
    "audioFormat": "mp3",
    "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
  }'
```

**Option C: View Interactive API Docs**
Open http://localhost:8000/docs in your browser

## 📡 API Endpoint

### Request

```
POST /api/voice-detection
```

**Headers:**
```
Content-Type: application/json
x-api-key: sk_test_123456789
```

**Body:**
```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "ID3BAAAAAAAjVFNTRVU..."
}
```

### Response

```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.91,
  "explanation": "Unnatural pitch consistency and robotic speech patterns detected"
}
```

## 🔐 Authentication

All requests require an API key in the `x-api-key` header:

```
x-api-key: sk_test_123456789
```

### Valid Test Keys
- `sk_test_123456789` (default)
- `your_secret_api_key`

To add more keys, edit `backend/main.py`:
```python
VALID_API_KEYS = {
    "sk_test_123456789",
    "your_secret_api_key",
    "new_api_key"  # Add here
}
```

## 📝 Request & Response Format

### Supported Languages

| Language | Code |
|----------|------|
| Tamil | `"Tamil"` |
| English | `"English"` |
| Hindi | `"Hindi"` |
| Malayalam | `"Malayalam"` |
| Telugu | `"Telugu"` |

### Classification Output

| Value | Meaning |
|-------|---------|
| `"AI_GENERATED"` | Voice was created using AI/synthetic systems |
| `"HUMAN"` | Voice was spoken by a real human |

### Audio Format

Only **MP3** format supported (`"mp3"`)

## 🧪 Testing with Python

### Convert MP3 to Base64

```python
import base64

with open("sample.mp3", "rb") as f:
    audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    
print(audio_base64)  # Use this in your request
```

### Test Request

```python
import requests
import base64

# Prepare audio
with open("sample.mp3", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode("utf-8")

# Send request
response = requests.post(
    "http://localhost:8000/api/voice-detection",
    json={
        "language": "Tamil",
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    },
    headers={"x-api-key": "sk_test_123456789"}
)

print(response.json())
```

## 📊 Classification Examples

### Example 1: AI-Generated Tamil Voice

**Input:** Tamil audio (synthetic TTS)
```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.89,
  "explanation": "Consistent pitch patterns and absence of natural speech variations detected. Phoneme transitions appear mechanical."
}
```

### Example 2: Human English Voice

**Input:** English audio (natural speaker)
```json
{
  "status": "success",
  "language": "English",
  "classification": "HUMAN",
  "confidenceScore": 0.92,
  "explanation": "Natural speech variations and authentic breathing patterns detected. Emotional expression and prosody indicate human speaker."
}
```

### Example 3: Uncertain Classification

**Input:** High-quality synthesized Hindi
```json
{
  "status": "success",
  "language": "Hindi",
  "classification": "HUMAN",
  "confidenceScore": 0.58,
  "explanation": "Audio characteristics partially match natural speech patterns. High-quality synthesis makes definitive classification difficult."
}
```

## ⚠️ Error Responses

### Missing API Key
```
HTTP 401
{
  "status": "error",
  "message": "Invalid API key or malformed request"
}
```

### Invalid Language
```
HTTP 400
{
  "status": "error",
  "message": "Unsupported language: Spanish. Supported: Tamil, English, Hindi, Malayalam, Telugu"
}
```

### Invalid Base64
```
HTTP 400
{
  "status": "error",
  "message": "Invalid Base64 encoding: ..."
}
```

### Corrupted Audio
```
HTTP 400
{
  "status": "error",
  "message": "Invalid audio file: ..."
}
```

## 🏗️ Project Structure

```
ai-audio-detector/
├── backend/
│   ├── main.py              # FastAPI application & endpoints
│   ├── models.py            # ML model loading & inference
│   ├── utils.py             # Audio preprocessing utilities
│   └── __init__.py
│
├── requirements.txt         # Python dependencies
├── test_voice_detection.py  # Test client for API
├── API_DOCUMENTATION.md     # Full API documentation
├── README.md               # This file
└── QUICKSTART.md           # Quick start guide
```

## 🤖 Models Used

### Language Detection
- **Model:** OpenAI Whisper-Small
- **Architecture:** Encoder-decoder transformer
- **Training Data:** 680k+ hours of multilingual audio
- **Languages Supported:** 99+

### AI Voice Detection
- **Model:** Audio Spectrogram Transformer (AST)
- **Task:** Audio classification
- **Training Data:** ASVspoof5 synthetic voice dataset
- **Features:** MFCC, Spectral characteristics, Energy patterns

## 🎓 Evaluation Criteria

According to the hackathon rules, we are evaluated on:

1. **🎯 Accuracy** - Correct AI vs Human detection across all 5 languages
2. **🌍 Language Consistency** - Works well across Tamil, English, Hindi, Malayalam, Telugu
3. **📦 Format Compliance** - Correct JSON request/response format
4. **⚡ Performance** - Fast response times, API reliability
5. **🧠 Explanation Quality** - Clear, detailed reasoning for classifications

## 🔧 Deployment

### Local Development
```bash
cd backend
python main.py
```

### With Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ ./backend/
EXPOSE 8000
CMD ["python", "backend/main.py"]
```

Build and run:
```bash
docker build -t ai-voice-detection .
docker run -p 8000:8000 ai-voice-detection
```

### Production Deployment

1. Change API keys in `backend/main.py`
2. Set `reload=False` in main.py
3. Use HTTPS/SSL
4. Configure CORS properly
5. Set up monitoring and logging
6. Use environment variables for secrets

## 🐛 Troubleshooting

### Issue: "Could not connect to localhost:8000"
**Solution:** Ensure server is running: `python backend/main.py`

### Issue: "Invalid API key or malformed request"
**Solution:** Check the `x-api-key` header value

### Issue: "Unsupported language"
**Solution:** Use exact language names: Tamil, English, Hindi, Malayalam, Telugu

### Issue: "Invalid Base64 encoding"
**Solution:** Ensure the audio file is being properly Base64 encoded

### Issue: Models taking too long to load
**Solution:** First run downloads 500MB of models. This is one-time only. Subsequent runs will be instant.

### Issue: Timeout on large files
**Solution:** Maximum tested file size is ~50MB. Split larger files.

## 📚 API Documentation

Full API documentation with examples is available at:

**After starting the server:**
- **Interactive Docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI Schema:** http://localhost:8000/openapi.json

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for complete details.

## ✅ Checklist for Hackathon Submission

- [ ] API running on localhost:8000 (or deployed URL)
- [ ] Authentication working with x-api-key header
- [ ] All 5 languages supported (Tamil, English, Hindi, Malayalam, Telugu)
- [ ] Correct response format (status, language, classification, confidenceScore, explanation)
- [ ] Base64 MP3 input working
- [ ] Error responses properly formatted
- [ ] API documentation available
- [ ] Test client working (test_voice_detection.py)
- [ ] Models downloaded and loaded successfully
- [ ] Response time < 30 seconds per request

## 📞 Support

For issues:
1. Check API logs on the terminal
2. Verify audio file exists and is valid MP3
3. Test with test_voice_detection.py script
4. Check that all dependencies are installed
5. Verify you have adequate disk space (2GB)

## 📅 Important Notes

**For Hackathon Evaluation:**
- Systems will send Base64 MP3 audio via POST /api/voice-detection
- Language will be one of the 5 supported languages
- API key will be provided separately
- Evaluation is based on accuracy, consistency, and format compliance

**Input Constraints:**
- Audio format: **MP3 only**
- Encoding: **Base64**
- Languages: **Fixed 5 languages only**
- Each request: **One audio file**
- No audio modification allowed

## 🎯 Next Steps

1. **Install dependencies:** `pip install -r requirements.txt`
2. **Start the server:** `python backend/main.py`
3. **Test with sample audio:** `python test_voice_detection.py sample.mp3 Tamil`
4. **Submit your API URL** to the evaluation system
5. **Monitor the responses** for accuracy and consistency

## 📄 Files Included

- `backend/main.py` - FastAPI application with /api/voice-detection endpoint
- `backend/models.py` - ML models and inference logic
- `backend/utils.py` - Audio preprocessing utilities
- `test_voice_detection.py` - Test client for easy API testing
- `requirements.txt` - Python dependencies
- `API_DOCUMENTATION.md` - Complete API documentation
- `QUICKSTART.md` - Quick start guide
- `README.md` - This file

---

**Built for HCL Hackthone 2026 - AI Audio Detector**

Good luck with your hackathon submission! 🚀
