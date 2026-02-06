import requests
import base64
import json
import sys

def test_voice_detection(api_url: str, audio_file_path: str, language: str = "en"):
    """
    Test the voice detection API with a local audio file
    
    Args:
        api_url: Full URL of the API endpoint (e.g., https://ai-audio-detector-xxxxx.run.app)
        audio_file_path: Path to your MP3 audio file
        language: Language code (en, hi, te, ta, ml)
    """
    
    # Read and encode audio file
    print(f"📁 Reading audio file: {audio_file_path}")
    with open(audio_file_path, 'rb') as f:
        audio_data = f.read()
    
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    print(f"✓ Encoded {len(audio_data)} bytes")
    
    # Prepare request
    endpoint = f"{api_url}/api/voice-detection"
    payload = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
    
    print(f"🚀 Sending request to: {endpoint}")
    print(f"   Language: {language}")
    print(f"   Audio size: {len(audio_base64)} characters")
    
    # Send request
    try:
        response = requests.post(endpoint, json=payload, timeout=300)
        response.raise_for_status()
        
        result = response.json()
        
        # Print formatted response
        print("\n✅ Response received:\n")
        print(json.dumps(result, indent=2))
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        sys.exit(1)


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python test_cloud_api.py <api_url> [audio_file] [language]")
        print("\nExamples:")
        print("  python test_cloud_api.py http://localhost:8000 test_audio.mp3")
        print("  python test_cloud_api.py https://ai-audio-detector-xxxxx.run.app test_audio.mp3 en")
        sys.exit(1)
    
    api_url = sys.argv[1]
    audio_file = sys.argv[2] if len(sys.argv) > 2 else "test_audio.mp3"
    language = sys.argv[3] if len(sys.argv) > 3 else "en"
    
    test_voice_detection(api_url, audio_file, language)
