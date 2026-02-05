"""
FastAPI Audio Analyzer Backend - AI Voice Detection API

This is the main entry point for the AI-powered audio analysis API.
Provides endpoints for detecting whether audio is AI-generated or human voice.

API Endpoints:
- POST /api/voice-detection - Analyze Base64 encoded MP3 audio
- GET / - Health check / API info
"""

import os
import base64
import io
import tempfile
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from utils import (
    preprocess_audio,
    cleanup_temp_file,
    validate_audio_file
)
from models import AudioAnalyzer


# Initialize FastAPI app
app = FastAPI(
    title="AI Voice Detection API",
    description="Detect whether voice is AI-generated or human across 5 languages",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key configuration
VALID_API_KEYS = {
    "ai-audio-detection-akatsukiapi",  # Main API key
    "your_secret_api_key"  # Add more keys as needed
}

# Global analyzer instance
analyzer: AudioAnalyzer = None


@app.on_event("startup")
async def startup_event():
    """Initialize ML models when the server starts."""
    global analyzer
    print("Starting up Audio Analyzer API...")
    print("Loading ML models (this may take a minute)...")
    
    try:
        analyzer = AudioAnalyzer()
        print("✓ Models loaded successfully!")
    except Exception as e:
        print(f"✗ Failed to load models: {str(e)}")
        print("Server will start but API endpoints will fail")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown."""
    print("Shutting down Audio Analyzer API...")


# Request/Response Models
class VoiceDetectionRequest(BaseModel):
    """Voice detection API request"""
    language: str  # Tamil, English, Hindi, Malayalam, Telugu
    audioFormat: str  # Always "mp3"
    audioBase64: str  # Base64 encoded MP3 audio


class VoiceDetectionResponse(BaseModel):
    """Voice detection API response"""
    status: str  # "success" or "error"
    language: str
    classification: str  # "AI_GENERATED" or "HUMAN"
    confidenceScore: float  # 0.0 to 1.0
    explanation: str


class ErrorResponse(BaseModel):
    """Error response"""
    status: str  # "error"
    message: str


# Routes

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "AI Voice Detection API",
        "version": "1.0.0",
        "status": "running",
        "supported_languages": [
            "Tamil", "English", "Hindi", "Malayalam", "Telugu"
        ],
        "endpoint": "/api/voice-detection",
        "authentication": "x-api-key header required"
    }


@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if analyzer is not None else "not loaded"
    
    return {
        "status": "healthy",
        "models": model_status
    }


@app.post("/api/voice-detection", response_model=VoiceDetectionResponse)
async def voice_detection(
    request: VoiceDetectionRequest,
    api_key: str = Header(None, alias="x-api-key")
):
    """
    Detect whether voice is AI-generated or human.
    
    Args:
        request: VoiceDetectionRequest containing:
            - language: Tamil, English, Hindi, Malayalam, Telugu
            - audioFormat: Always "mp3"
            - audioBase64: Base64 encoded MP3 audio
        api_key: API key from x-api-key header
    
    Returns:
        VoiceDetectionResponse with classification and confidence score
    
    Raises:
        HTTPException: If validation or processing fails
    """
    
    # Validate API key
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key or malformed request"
        )
    
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key or malformed request"
        )
    
    # Check if models are loaded
    if analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="ML models not loaded. Please wait for server initialization."
        )
    
    # Validate request fields
    if not request.language:
        raise HTTPException(
            status_code=400,
            detail="Language field is required"
        )
    
    if not request.audioBase64:
        raise HTTPException(
            status_code=400,
            detail="audioBase64 field is required"
        )
    
    # Validate language
    supported_languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    if request.language not in supported_languages:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: {request.language}. Supported: {', '.join(supported_languages)}"
        )
    
    # Validate audio format
    if request.audioFormat.lower() != "mp3":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {request.audioFormat}. Only 'mp3' is supported."
        )
    
    temp_input_path = None
    
    try:
        # Decode Base64 MP3
        print(f"Decoding Base64 MP3 audio...")
        try:
            audio_bytes = base64.b64decode(request.audioBase64)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid Base64 encoding: {str(e)}"
            )
        
        # Save to temporary MP3 file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_input_path = temp_file.name
            temp_file.write(audio_bytes)
        
        # Validate audio file
        is_valid, error_msg = validate_audio_file(temp_input_path)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid audio file: {error_msg}"
            )
        
        # Preprocess audio (convert to mono, resample to 16kHz)
        print(f"Processing audio...")
        audio_data, sample_rate = preprocess_audio(temp_input_path)
        
        # Run analysis
        print("Running audio analysis...")
        result = analyzer.analyze(audio_data, sample_rate)
        
        # Extract AI/REAL scores
        ai_score = 0.0
        real_score = 0.0
        for score_item in result["ai_voice_result"]:
            if score_item["label"] == "AI":
                ai_score = score_item["score"]
            elif score_item["label"] == "REAL":
                real_score = score_item["score"]
        
        # Determine classification
        if ai_score > real_score:
            classification = "AI_GENERATED"
            confidence = ai_score
            explanation = analyzer.get_ai_explanation(ai_score, result["language_detected"])
        else:
            classification = "HUMAN"
            confidence = real_score
            explanation = analyzer.get_human_explanation(real_score, result["language_detected"])
        
        # Format response
        response = {
            "status": "success",
            "language": request.language,
            "classification": classification,
            "confidenceScore": round(confidence, 2),
            "explanation": explanation
        }
        
        print(f"Analysis complete: {request.language} -> {classification}")
        return JSONResponse(content=response)
    
    except HTTPException:
        raise
    
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )
    
    finally:
        # Cleanup temporary files
        if temp_input_path:
            cleanup_temp_file(temp_input_path)


# Main entry point
if __name__ == "__main__":
    """
    Run the FastAPI server using uvicorn.
    
    Usage:
        python main.py
    
    Server will start on http://localhost:8000
    """
    print("=" * 60)
    print("AI Voice Detection API")
    print("=" * 60)
    print("\nStarting server...")
    print("API Documentation: http://localhost:8000/docs")
    print("OpenAPI Schema: http://localhost:8000/openapi.json")
    print("\nSupported Languages: Tamil, English, Hindi, Malayalam, Telugu")
    print("Authentication: x-api-key header (ai-audio-detection-akatsukiapi)")
    print("\nPress CTRL+C to stop the server\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
