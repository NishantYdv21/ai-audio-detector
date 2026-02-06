# Memory Optimization Troubleshooting

## Quick Fixes for OOM Errors on Render Free Tier

### Status: Still Getting OOM?
If you're still seeing "Ran out of memory" errors after deploying the optimized version, use these escalating solutions:

---

## Level 1: Enable 8-Bit Quantization (Easiest)

**Time to implement**: 2 minutes

1. Go to your Render dashboard
2. Navigate to your web service → Environment
3. Add new environment variable:
   ```
   Key:   USE_INT8_QUANTIZATION
   Value: true
   ```
4. Redeploy

**Result**: Models load 4x faster in memory, slower inference (2-3s instead of 1s)

First, install bitsandbytes in requirements:
```bash
bitsandbytes>=0.41.0
```

---

## Level 2: Skip Language Detection (Medium Impact)

**Time to implement**: 5 minutes
**Memory saved**: ~300MB

Edit `backend/models.py` in the `detect_language()` method (around line ~150):

Replace the entire method with:
```python
def detect_language(self, audio: np.ndarray, sr: int) -> Dict:
    """
    Simplified language detection - returns English as default.
    Full Whisper model removed to save memory.
    Users can specify language in API request if needed.
    """
    return {
        'language_code': 'en',
        'language_name': 'English',
        'confidence': 1.0  # Assume English
    }
```

**Result**: Saves 300MB, eliminates language detection capability

---

## Level 3: Use Ultra-Lightweight Model (Most Aggressive)

**Time to implement**: 10 minutes
**Memory saved**: ~150MB additional

Replace whisper-tiny with wav2vec2 for language-agnostic detection:

In `backend/models.py`, replace `_ensure_language_model_loaded()`:
```python
def _ensure_language_model_loaded(self):
    """Skip language detection - not needed"""
    if self._whisper_model is not None:
        return
    print("  Skipping language detection (not loaded)")
    # Don't load any model
```

**Result**: Only one model (AI detection) loaded, uses ~200MB total

---

## Level 4: Using Distilled Models (Advanced)

**Time to implement**: 15 minutes
**Memory saved**: ~100MB more

Use distilled (smaller) versions:
```python
# In _ensure_ai_detection_model_loaded(), try these lighter models first:
model_options = [
    "facebook/wav2vec2-base-superb-ks",  # Smaller ASVspoof alternative
    "MattyB95/AST-ASVspoof5-Synthetic-Voice-Detection",
]
```

---

## Level 5: Upgrade from Free Tier (Nuclear Option)

**Cost**: $7/month (Starter plan)
**Time to implement**: 30 seconds

If optimizations don't work, consider upgrading:

Render Pricing:
- **Free**: 512MB RAM (problematic)
- **Starter**: 512MB RAM + reserved instance ($7/month)
- **Standard**: 2GB+ RAM ($20+/month)

The Starter plan guarantees your instance stays running (no auto-sleep) and has faster cold starts.

---

## Recommended Approach by Use Case

### If you ONLY need AI voice detection:
→ Use **Level 2** (skip language detection)
- Memory usage: ~250MB
- Simplest solution, most reliable

### If you need language detection:
→ Use **Level 1** (8-bit quantization)
- Memory usage: ~400MB at peak
- Some performance trade-off, but works

### If you can't get it working:
→ Use **Level 5** (upgrade to Starter)
- Most reliable solution
- Guaranteed 512MB RAM consistently
- Faster deployments

---

## Testing Memory Usage Locally

Before deploying, test your memory usage locally:

```bash
# Run with memory monitoring
python -c "
import os
import psutil

process = psutil.Process(os.getpid())
print(f'Initial memory: {process.memory_info().rss / 1024 / 1024:.1f}MB')

# Import and initialize
from backend.models import AudioAnalyzer
analyzer = AudioAnalyzer()
print(f'After init: {process.memory_info().rss / 1024 / 1024:.1f}MB')

# Trigger model loading with dummy audio
import numpy as np
dummy_audio = np.random.randn(16000 * 10).astype(np.float32)
print(f'Before inference: {process.memory_info().rss / 1024 / 1024:.1f}MB')
analyzer.detect_ai_voice(dummy_audio, 16000)
print(f'After inference: {process.memory_info().rss / 1024 / 1024:.1f}MB')
"
```

---

## Environment Variables Reference

These are automatically set in Dockerfile.render, but you can override in Render dashboard:

- `PYTHONUNBUFFERED=1` - Don't buffer Python output
- `PYTHONDONTWRITEBYTECODE=1` - Don't create .pyc files
- `OMP_NUM_THREADS=1` - Limit OpenMP threads
- `MKL_NUM_THREADS=1` - Limit MKL threads
- `TOKENIZERS_PARALLELISM=false` - Disable parallel tokenization
- `USE_INT8_QUANTIZATION=true` - Enable 8-bit model quantization

---

## Render Documentation Links

- [Render Free Tier Limits](https://render.com/docs/free)
- [Render Environment Variables](https://render.com/docs/environment-variables)
- [Render Python Deployments](https://render.com/docs/deploy-python)

---

## Support

If none of these work:
1. Check Render logs for actual error message
2. Try Level 2 (skip language detection) - most reliable
3. Consider paid tier - models need more than 512MB of guaranteed memory
