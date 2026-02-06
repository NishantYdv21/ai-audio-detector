#!/usr/bin/env python3
"""
HuggingFace Spaces compatible entry point
This allows deployment without Google Cloud SDK
"""

import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.main import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))  # HF Spaces uses 7860
    uvicorn.run(app, host="0.0.0.0", port=port)
