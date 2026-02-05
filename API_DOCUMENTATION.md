# AI Voice Detection API - Hackathon Implementation

## Overview

This API detects whether a voice sample is **AI-Generated** or **Human** across 5 supported languages: Tamil, English, Hindi, Malayalam, and Telugu.

## Supported Languages

- ✅ Tamil
- ✅ English  
- ✅ Hindi
- ✅ Malayalam
- ✅ Telugu

## API Endpoint

```
POST https://your-domain.com/api/voice-detection
```

**Local Development:**
```
POST http://localhost:8000/api/voice-detection
```

## Authentication

All requests must include the API key in the request header:

```
x-api-key: sk_test_123456789
```

**Error Response (Invalid API Key):**
```json
{
  "status": "error",
  "message": "Invalid API key or malformed request"
}
```

## Request Format

### Request Body (JSON)

```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "ID3BAAAAAAAjVFNTRUAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
}
```

### Request Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `language` | String | Language of the audio file | `"Tamil"`, `"English"`, `"Hindi"`, `"Malayalam"`, `"Telugu"` |
| `audioFormat` | String | Audio format (always mp3) | `"mp3"` |
| `audioBase64` | String | Base64 encoded MP3 audio data | `"SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."` |

## Response Format

### Success Response (HTTP 200)

```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.91,
  "explanation": "Unnatural pitch consistency and robotic speech patterns detected"
}
```

### Response Fields

| Field | Type | Description | Values |
|-------|------|-------------|--------|
| `status` | String | Request status | `"success"`, `"error"` |
| `language` | String | Language of the audio | `"Tamil"`, `"English"`, `"Hindi"`, `"Malayalam"`, `"Telugu"` |
| `classification` | String | Voice classification | `"AI_GENERATED"`, `"HUMAN"` |
| `confidenceScore` | Float | Confidence (0.0 to 1.0) | `0.0` to `1.0` |
| `explanation` | String | Brief reason for classification | Human-readable text |

### Error Response (HTTP 4xx/5xx)

```json
{
  "status": "error",
  "message": "Invalid API key or malformed request"
}
```

## cURL Examples

### Test Request - AI Generated Voice

```bash
curl -X POST http://localhost:8000/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_test_123456789" \
  -d '{
    "language": "Tamil",
    "audioFormat": "mp3",
    "audioBase64": "ID3BAAAAAAAjVFNTRVU..."
  }'
```

### Test Request - Human Voice

```bash
curl -X POST http://localhost:8000/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_test_123456789" \
  -d '{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "ID3BAAAAAAAjVFNTRVU..."
  }'
```

## Python Test Client Usage

### Installation

```bash
pip install requests
```

### Usage

```bash
# Basic usage (uses default API key)
python test_voice_detection.py sample.mp3 Tamil

# With custom API key
python test_voice_detection.py sample.mp3 English sk_test_123456789

# Other languages
python test_voice_detection.py sample.mp3 Hindi
python test_voice_detection.py sample.mp3 Malayalam
python test_voice_detection.py sample.mp3 Telugu
```

### Python Script Example

```python
import requests
import base64

# Read MP3 file
with open("sample.mp3", "rb") as f:
    audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

# Prepare request
url = "http://localhost:8000/api/voice-detection"
headers = {
    "Content-Type": "application/json",
    "x-api-key": "sk_test_123456789"
}

payload = {
    "language": "Tamil",
    "audioFormat": "mp3",
    "audioBase64": audio_base64
}

# Send request
response = requests.post(url, json=payload, headers=headers)
result = response.json()

print(result)
# Output:
# {
#   "status": "success",
#   "language": "Tamil",
#   "classification": "AI_GENERATED",
#   "confidenceScore": 0.91,
#   "explanation": "..."
# }
```

## Error Handling

### Missing API Key
```json
{
  "status": "error",
  "message": "Invalid API key or malformed request"
}
```

### Invalid Language
```json
{
  "status": "error",
  "message": "Unsupported language: Spanish. Supported: Tamil, English, Hindi, Malayalam, Telugu"
}
```

### Invalid Audio Format
```json
{
  "status": "error",
  "message": "Unsupported audio format: wav. Only 'mp3' is supported."
}
```

### Invalid Base64
```json
{
  "status": "error",
  "message": "Invalid Base64 encoding: ..."
}
```

### Server Error
```json
{
  "status": "error",
  "message": "ML models not loaded. Please wait for server initialization."
}
```

## Running the API Server

### Prerequisites

- Python 3.8+
- Dependencies installed: `pip install -r requirements.txt`

### Start Server

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

### Server Configuration

- **Host:** 0.0.0.0 (accessible from any IP)
- **Port:** 8000
- **Reload:** Enabled for development
- **Log Level:** Info

## API Documentation

Interactive API documentation available at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI Schema:** http://localhost:8000/openapi.json

## Classification Logic

### AI_GENERATED
Detected when:
- Unnatural pitch consistency
- Robotic speech patterns
- Lack of breathing sounds
- Consistent prosody
- Minimal speech artifacts
- Regular phoneme transitions

### HUMAN
Detected when:
- Natural speech variations
- Authentic breathing patterns
- Organic phoneme transitions
- Natural prosody
- Realistic speech artifacts
- Genuine emotional expression

## Confidence Score Interpretation

| Score Range | Interpretation |
|-------------|-----------------|
| 0.90 - 1.00 | **Very High** - Strong indicators of classification |
| 0.75 - 0.89 | **High** - Multiple signs confirm classification |
| 0.60 - 0.74 | **Moderate** - Reasonable confidence in classification |
| 0.50 - 0.59 | **Low** - Weak indicators, close decision |
| < 0.50 | **Uncertain** - Difficult to classify |

## Request Constraints

- Audio format: **MP3 only**
- Maximum file size: **~50MB** (Base64 encoded)
- Audio duration tested: **up to 5 minutes**
- Languages: **Fixed 5 languages only**
- One audio per request

## Health Check Endpoint

```bash
curl http://localhost:8000/health/
```

Response:
```json
{
  "status": "healthy",
  "models": "loaded"
}
```

## API Key Management

### Valid API Keys (for testing)

- `sk_test_123456789` (default test key)
- `your_secret_api_key`

### Adding New API Keys

Edit `backend/main.py` and update the `VALID_API_KEYS` set:

```python
VALID_API_KEYS = {
    "sk_test_123456789",
    "your_secret_api_key",
    "another_api_key"  # Add more keys
}
```

## Model Information

### Language Detection
- **Model:** OpenAI Whisper-Small
- **Training Data:** 680k+ hours of multilingual audio
- **Languages:** 99+ languages supported
- **Architecture:** Encoder-decoder transformer

### AI Voice Detection
- **Model:** Audio Spectrogram Transformer (AST)
- **Training Data:** ASVspoof5 synthetic voice dataset
- **Features:** MFCC, Spectral features, Energy analysis
- **Architecture:** Transformer-based audio classification

## Deployment

### Production Checklist

- [ ] Change default API keys (`VALID_API_KEYS`)
- [ ] Enable HTTPS/SSL
- [ ] Set `reload=False` in uvicorn configuration
- [ ] Configure allowed CORS origins
- [ ] Use environment variables for API keys
- [ ] Monitor model loading on startup
- [ ] Set up logging and monitoring
- [ ] Configure rate limiting
- [ ] Test with real microphone recordings

### Docker Deployment

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

## Troubleshooting

### API Connection Failed
```
Error: Could not connect to http://localhost:8000
```
**Solution:** Ensure the server is running with `python backend/main.py`

### Invalid API Key
```
{
  "status": "error",
  "message": "Invalid API key or malformed request"
}
```
**Solution:** Check the `x-api-key` header value

### Timeout Error
```
Error: Request timeout (30s)
```
**Solution:** The audio file might be corrupted or too large. Verify the MP3 file.

### Models Not Loaded
```
{
  "status": "error",
  "message": "ML models not loaded..."
}
```
**Solution:** Wait for server to fully initialize (may take 1-2 minutes on first run)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the API logs on the server terminal
3. Verify the MP3 file format and size
4. Test with `test_voice_detection.py` script

## License

This project is part of HCL Hackthone 2026 - AI Audio Detector
