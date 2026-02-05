#!/usr/bin/env python3
"""
Test client for AI Voice Detection API

Usage:
    python test_voice_detection.py <mp3_file_path> <language> [api_key]

Examples:
    python test_voice_detection.py sample.mp3 Tamil
    python test_voice_detection.py sample.mp3 English sk_test_123456789
    python test_voice_detection.py sample.mp3 Hindi your_secret_api_key
"""

import requests
import base64
import sys
import json
from pathlib import Path


def test_voice_detection(mp3_file_path, language, api_key="ai-audio-detection-akatsukiapi", base_url="http://localhost:8000"):
    """
    Test the voice detection API with a Base64 encoded MP3 file
    
    Args:
        mp3_file_path: Path to MP3 file
        language: Language name (Tamil, English, Hindi, Malayalam, Telugu)
        api_key: API key for authentication
        base_url: Base URL of the API (default: localhost:8000)
    
    Returns:
        Response JSON or None if error
    """
    
    # Validate file exists
    mp3_path = Path(mp3_file_path)
    if not mp3_path.exists():
        print(f"❌ Error: File not found: {mp3_file_path}")
        return None
    
    # Validate language
    supported_languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    if language not in supported_languages:
        print(f"❌ Error: Unsupported language '{language}'")
        print(f"   Supported: {', '.join(supported_languages)}")
        return None
    
    # Read and encode MP3 file as Base64
    print(f"📄 Reading MP3 file: {mp3_file_path}")
    with open(mp3_path, "rb") as f:
        audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    
    print(f"✓ File size: {len(audio_bytes) / 1024:.2f} KB")
    print(f"✓ Base64 encoded: {len(audio_base64)} characters")
    
    # Prepare API request
    url = f"{base_url}/api/voice-detection"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key
    }
    
    payload = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
    
    print(f"\n📤 Sending request to {url}")
    print(f"   Language: {language}")
    print(f"   API Key: {api_key[:10]}...")
    
    try:
        # Send request
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        
        # Check response status
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Success! Response:")
            print(json.dumps(result, indent=2))
            return result
        else:
            print(f"\n❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return None
    
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Error: Could not connect to {base_url}")
        print(f"Make sure the API server is running: python backend/main.py")
        return None
    
    except requests.exceptions.Timeout:
        print(f"\n❌ Error: Request timeout (30s)")
        print(f"The audio file may be too large or processing is slow")
        return None
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return None


def main():
    """Main entry point"""
    if len(sys.argv) < 3:
        print("Usage: python test_voice_detection.py <mp3_file> <language> [api_key]")
        print("\nExamples:")
        print("  python test_voice_detection.py sample.mp3 Tamil")
        print("  python test_voice_detection.py sample.mp3 English ai-audio-detection-akatsukiapi")
        print("\nSupported languages: Tamil, English, Hindi, Malayalam, Telugu")
        sys.exit(1)
    
    mp3_file = sys.argv[1]
    language = sys.argv[2]
    api_key = sys.argv[3] if len(sys.argv) > 3 else "ai-audio-detection-akatsukiapi"
    
    print("=" * 60)
    print("AI Voice Detection API - Test Client")
    print("=" * 60)
    print()
    
    test_voice_detection(mp3_file, language, api_key)


if __name__ == "__main__":
    main()
