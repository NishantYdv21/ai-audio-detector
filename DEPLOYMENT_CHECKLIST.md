# Deployment Checklist

Before deploying to Render, verify all these changes are in place:

## Code Changes ✓

- [ ] `backend/models.py` 
  - [ ] Uses `whisper-tiny` instead of `whisper-small`
  - [ ] Has lazy loading with `_ensure_language_model_loaded()` 
  - [ ] Has lazy loading with `_ensure_ai_detection_model_loaded()`
  - [ ] Uses properties for model access
  - [ ] Supports `USE_INT8_QUANTIZATION` environment variable
  - [ ] Memory optimization flags set at file top

- [ ] `backend/main.py`
  - [ ] Updated startup event mentions "lazy loading"
  - [ ] No model loading in `startup_event()`

## Configuration Files ✓

- [ ] `requirements-optimized.txt` exists with:
  - [ ] Lightweight dependencies
  - [ ] torch, torchaudio, transformers listed
  - [ ] No unnecessary packages

- [ ] `Dockerfile.render` exists with:
  - [ ] `FROM python:3.10-slim` (lightweight base)
  - [ ] Memory optimization ENV variables
  - [ ] `uvicorn backend.main:app --host 0.0.0.0 --port 8080` in CMD
  - [ ] Single worker configuration

- [ ] `render.yaml` exists with:
  - [ ] `buildCommand: pip install --no-cache-dir -r requirements-optimized.txt`
  - [ ] `startCommand: uvicorn backend.main:app --host 0.0.0.0 --port 8080`
  - [ ] Memory optimization environment variables

## Documentation ✓

- [ ] `RENDER_DEPLOYMENT.md` - Deployment guide
- [ ] `TROUBLESHOOTING_OOM.md` - Escalating solutions
- [ ] `OPTIMIZATION_SUMMARY.md` - What changed and why

## Git Status ✓

```bash
git status
# Should show these files as new or modified:
# ✓ backend/models.py (MODIFIED)
# ✓ backend/main.py (MODIFIED)
# ✓ requirements-optimized.txt (NEW)
# ✓ Dockerfile.render (NEW)
# ✓ render.yaml (NEW)
# ✓ RENDER_DEPLOYMENT.md (NEW)
# ✓ TROUBLESHOOTING_OOM.md (NEW)
# ✓ OPTIMIZATION_SUMMARY.md (NEW)
```

## Ready to Deploy?

### Step 1: Commit Changes
```bash
git add .
git commit -m "Optimize for Render free tier: lazy loading + whisper-tiny"
git push
```

### Step 2: Configure on Render
1. Go to render.com → Dashboard
2. Create new Web Service (or update existing)
3. Connect your GitHub repo
4. **Name**: `ai-audio-detector` (or your app name)
5. **Build Command**: 
   ```
   pip install --no-cache-dir -r requirements-optimized.txt
   ```
6. **Start Command**: 
   ```
   uvicorn backend.main:app --host 0.0.0.0 --port 8080
   ```
7. **Plan**: Free
8. Click "Create Web Service"

### Step 3: Monitor Deployment
- Watch logs for: `✓ Audio Analyzer initialized`
- First API request will take 3-5 seconds (model loading)
- Subsequent requests will be fast

### Step 4: Test API
Call your deployed endpoint:
```bash
curl https://your-app.onrender.com/
```

Should return API info (not OOM error).

## If OOM Still Occurs

### Quick Fix #1: 8-Bit Quantization
1. Go to Render Dashboard → Environment
2. Add: `USE_INT8_QUANTIZATION=true`
3. Click "Save"
4. Redeploy

### Quick Fix #2: Skip Language Detection  
See `TROUBLESHOOTING_OOM.md` for detailed instructions

### Quick Fix #3: Upgrade Plan
Switch to Starter ($7/month) for guaranteed memory.

---

## Expected Timeline

| Stage | Time | Status |
|-------|------|--------|
| Push to GitHub | 10s | ✓ Instant |
| Render detects changes | 30s | ✓ Auto |
| Dependency install | 2-3min | ⏳ In progress |
| App startup | 30s | ✓ Quick |
| **First API request** | 3-5s | **⏳ Model loading** |
| Subsequent requests | <500ms | ✓ Fast |

---

## Verification Checklist After Deployment

Once deployed, verify:

- [ ] App shows as "Live" on Render dashboard (green checkmark)
- [ ] Logs show "*✓* Audio Analyzer initialized successfully!"
- [ ] Can access root endpoint: `GET https://your-app.onrender.com/`
- [ ] First request takes 3-5 seconds
- [ ] Second request is much faster (<1 second)
- [ ] Memory stays under 512MB in Render metrics
- [ ] No "Ran out of memory" errors in logs

---

## Need Help?

1. **OOM Errors?** → Read `TROUBLESHOOTING_OOM.md`
2. **Deployment Issues?** → Read `RENDER_DEPLOYMENT.md`
3. **Questions about changes?** → Read `OPTIMIZATION_SUMMARY.md`

---

## Key Metrics

After deployment, check Render dashboard:

- **Memory Usage**: Should be 400-420MB peak (not 900MB+)
- **CPU Usage**: Should be normal
- **Error Rate**: Should be 0%
- **Response Time**: First request 3-5s, others <1s

✅ **Ready to deploy!**
