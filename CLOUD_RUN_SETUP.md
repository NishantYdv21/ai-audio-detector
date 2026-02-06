## 🚀 Deploy to Google Cloud Run - Step by Step

### **Prerequisites (5 min setup)**
1. Create free Google Cloud account: https://cloud.google.com/free
2. Install Google Cloud CLI: https://cloud.google.com/sdk/docs/install
3. Login: `gcloud auth login`
4. Set project: `gcloud config set project YOUR_PROJECT_ID`

---

### **Step 1: Deploy to Cloud Run**

```bash
gcloud run deploy ai-audio-detector \
  --source . \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300 \
  --allow-unauthenticated
```

Takes ~5-10 min. You'll get a URL like:
```
https://ai-audio-detector-xxxxx.run.app
```

---

### **Step 2: Test with JSON (from Windows PowerShell)**

Save this as `test_cloud.ps1`:

```powershell
# Convert MP3 to Base64
$audioPath = "C:\path\to\your\audio.mp3"
$audioBase64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($audioPath))

# Prepare JSON
$body = @{
    language = "en"
    audioFormat = "mp3"
    audioBase64 = $audioBase64
} | ConvertTo-Json

# Send request
$url = "https://ai-audio-detector-xxxxx.run.app/api/voice-detection"
$response = Invoke-RestMethod -Uri $url -Method Post -Body $body -ContentType "application/json"

# Print response
$response | ConvertTo-Json | Write-Host
```

Run it:
```powershell
.\test_cloud.ps1
```

---

### **Step 3: Response Example**

```json
{
  "status": "success",
  "language": "en",
  "classification": "HUMAN",
  "confidenceScore": 0.92,
  "explanation": "Voice pattern matches human characteristics"
}
```

---

### **Cost (Free Tier)**
- ✅ 2 million requests/month FREE
- ✅ 360,000 compute seconds (vCPU × seconds)
- ✅ Each request ~60 seconds = ~6000 requests free

**After free tier**: ~$0.00002 per second (very cheap for hackathon)

---

### **Troubleshooting**

**Error: "Out of memory"**
- Increase `--memory` flag to 4Gi or 8Gi

**Error: "Cold start timeout"**
- First request takes 30-60s (normal), subsequent requests are fast

**Large model files**
- Models are downloaded on first run (takes time but cached)

---

### **Stop/Delete Deployment**
```bash
gcloud run delete ai-audio-detector
```
