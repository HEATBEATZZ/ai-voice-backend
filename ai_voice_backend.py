from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import time
import threading
import tempfile
import base64
from datetime import datetime
import speech_recognition as sr
from groq import Groq
from a4f_local import A4F
import io
from pydub import AudioSegment
from pydub.playback import play
import pygame
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sys
import subprocess
import atexit

# Check for ffmpeg and warn if not found
try:
    # Try to find ffmpeg in PATH
    ffmpeg_path = None
    for path in os.environ["PATH"].split(os.pathsep):
        if os.path.exists(os.path.join(path, "ffmpeg.exe")):
            ffmpeg_path = os.path.join(path, "ffmpeg.exe")
            break
    
    if not ffmpeg_path:
        # Try common installation locations on Windows
        common_paths = [
            os.path.join(os.environ.get('PROGRAMFILES', 'C:\\Program Files'), 'ffmpeg', 'bin', 'ffmpeg.exe'),
            os.path.join(os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)'), 'ffmpeg', 'bin', 'ffmpeg.exe'),
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'ffmpeg', 'bin', 'ffmpeg.exe')
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                ffmpeg_path = path
                break
    
    if not ffmpeg_path:
        print("⚠️ WARNING: ffmpeg not found in PATH or common locations. Audio processing may not work correctly.")
        print("Please install ffmpeg and add it to your PATH.")
        print("You can download it from: https://ffmpeg.org/download.html")
        print("After installation, make sure to add the ffmpeg bin directory to your system PATH.")
    else:
        print(f"✅ Found ffmpeg at: {ffmpeg_path}")
except Exception as e:
    print(f"Error checking for ffmpeg: {e}")

app = Flask(__name__)
CORS(app)

# Initialize services
groq_client = Groq(api_key="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")
tts_client = A4F()

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Global storage
conversation_history = []
current_session = None
silence_timer = None
executor = ThreadPoolExecutor(max_workers=4)

# Add session management
SESSIONS = {}
SESSION_TIMEOUT = 3600  # 1 hour in seconds

def cleanup_expired_sessions():
    """Clean up expired sessions"""
    current_time = time.time()
    expired_sessions = []
    
    for session_id, session in SESSIONS.items():
        if current_time - session.start_time.timestamp() > SESSION_TIMEOUT:
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        cleanup_session(session_id)

def cleanup_session(session_id):
    """Clean up a specific session and its resources"""
    if session_id in SESSIONS:
        session = SESSIONS[session_id]
        # Clean up audio files
        for message in session.messages:
            if message.get("audio_file") and os.path.exists(message["audio_file"]):
                safe_delete_file(message["audio_file"])
        # Remove session
        del SESSIONS[session_id]

class ConversationSession:
    def __init__(self):
        self.id = str(int(time.time()))
        self.start_time = datetime.now()
        self.messages = []
        self.current_input = ""
        self.is_listening = False
        self.last_speech_time = None
        self.silence_threshold = 2.0  # 2 seconds
        
    def add_message(self, role, content, language="auto", audio_file=None):
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "language": language,
            "audio_file": audio_file
        }
        self.messages.append(message)
        return message

# Language detection helper
def detect_language(text):
    """Simple language detection based on character sets"""
    hindi_chars = set('अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशसहक्षत्रज्ञ')
    text_chars = set(text)
    
    if hindi_chars.intersection(text_chars):
        return "hi"
    return "en"

# AI Processing
def process_with_groq(user_input, conversation_context=None):
    """Process user input with Groq AI"""
    try:
        language = detect_language(user_input)
        
        # Build context-aware system message
        system_message = {
            "role": "system",
            "content": """You are a helpful, Female conversational AI assistant name Rizza, Made by Ashmit Singh that can speak both Hindi and English. 
            - Respond in the same language the user speaks
            - If user speaks Hindi, respond in Hindi (Devanagari script)
            - If user speaks English, respond in English
            - Keep responses conversational and natural
            - Be helpful and engaging
            - For mixed language input, respond in the dominant language used"""
        }
        
        messages = [system_message]
        
        # Add conversation context
        if conversation_context:
            messages.extend(conversation_context[-6:])  # Last 6 messages for context
            
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="gemma2-9b-it",
            temperature=0.7,
            max_tokens=1000
        )
        
        ai_response = chat_completion.choices[0].message.content
        response_language = detect_language(ai_response)
        
        return {
            "response": ai_response,
            "language": response_language,
            "input_language": language
        }
        
    except Exception as e:
        print(f"Groq API error: {e}")
        return {
            "response": "I'm sorry, I encountered an error processing your request.",
            "language": "en",
            "input_language": "en"
        }

# TTS Processing
def generate_speech(text, language="en", voice="nova"):
    """Generate speech using a4f-local TTS"""
    try:
        # Generate audio bytes
        audio_bytes = tts_client.audio.speech.create(
            model="tts-1",
            input=text,
            voice=voice
        )
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.write(audio_bytes)
        temp_file.close()
        
        return temp_file.name
    
    except Exception as e:
        print(f"TTS error: {e}")
        return None

def cleanup_temp_files():
    """Clean up temporary audio files"""
    try:
        temp_dir = tempfile.gettempdir()
        for filename in os.listdir(temp_dir):
            if filename.endswith(('.wav', '.mp3')) and filename.startswith('tmp'):
                try:
                    file_path = os.path.join(temp_dir, filename)
                    # Check if file is older than 1 hour
                    if time.time() - os.path.getctime(file_path) > 3600:
                        try:
                            os.unlink(file_path)
                        except PermissionError:
                            # File might be in use, skip it
                            continue
                        except Exception as e:
                            print(f"Error deleting file {file_path}: {e}")
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
    except Exception as e:
        print(f"Error during cleanup: {e}")

def safe_delete_file(file_path):
    """Safely delete a file with retries"""
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
            return True
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            print(f"Could not delete file {file_path} after {max_retries} attempts")
            return False
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
            return False

# Speech Recognition
def transcribe_audio(audio_file):
    """Transcribe audio file to text"""
    recognizer = sr.Recognizer()
    wav_file = None
    
    try:
        # Convert audio to WAV if needed
        audio = AudioSegment.from_file(audio_file)
        wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio.export(wav_file.name, format="wav")
        
        with sr.AudioFile(wav_file.name) as source:
            audio_data = recognizer.record(source)
        
        # Try Hindi first, then English
        try:
            text = recognizer.recognize_google(audio_data, language="hi-IN")
            language = "hi"
        except sr.UnknownValueError:
            try:
                text = recognizer.recognize_google(audio_data, language="en-US")
                language = "en"
            except sr.UnknownValueError:
                return None, None
        
        return text, language
    
    except Exception as e:
        print(f"Speech recognition error: {e}")
        return None, None
    
    finally:
        if wav_file:
            safe_delete_file(wav_file.name)

# Silence detection and processing
def handle_silence_timeout():
    """Process accumulated speech after silence timeout"""
    global current_session, silence_timer
    
    if current_session and current_session.current_input.strip():
        # Process the accumulated input
        user_input = current_session.current_input.strip()
        current_session.current_input = ""
        
        # Add user message to session
        user_msg = current_session.add_message("user", user_input)
        
        # Get conversation context
        context = [{"role": msg["role"], "content": msg["content"]} 
                  for msg in current_session.messages[-10:]]
        
        # Process with AI
        ai_result = process_with_groq(user_input, context)
        
        # Add AI response to session
        ai_msg = current_session.add_message("assistant", ai_result["response"], ai_result["language"])
        
        # Generate TTS
        voice = "nova" if ai_result["language"] == "en" else "alloy"  # Different voices for different languages
        audio_file = generate_speech(ai_result["response"], ai_result["language"], voice)
        
        if audio_file:
            ai_msg["audio_file"] = audio_file
        
        print(f"User: {user_input}")
        print(f"AI ({ai_result['language']}): {ai_result['response']}")

# API Endpoints
@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/start_session', methods=['POST'])
def start_session():
    global current_session
    
    # Clean up old session if exists
    if current_session:
        cleanup_session(current_session.id)
    
    # Create new session
    current_session = ConversationSession()
    SESSIONS[current_session.id] = current_session
    
    return jsonify({
        "status": "success",
        "session_id": current_session.id,
        "message": "New conversation session started"
    })

@app.route('/transcribe', methods=['POST'])
def transcribe():
    global current_session, silence_timer
    
    if not current_session:
        return jsonify({"error": "No active session"}), 400
    
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    if not audio_file.filename:
        return jsonify({"error": "Empty audio file"}), 400
    
    try:
        # Save the uploaded file temporarily
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio_file.save(temp_audio.name)
        temp_audio.close()
        
        # Transcribe the audio
        text, language = transcribe_audio(temp_audio.name)
        
        # Clean up the temporary file
        safe_delete_file(temp_audio.name)
        
        if text is None:
            return jsonify({"error": "Could not transcribe audio"}), 400
        
        # Update session
        current_session.current_input += " " + text
        current_session.last_speech_time = time.time()
        
        # Reset silence timer
        if silence_timer:
            silence_timer.cancel()
        silence_timer = threading.Timer(current_session.silence_threshold, handle_silence_timeout)
        silence_timer.start()
        
        return jsonify({
            "status": "success",
            "text": text,
            "language": language
        })
        
    except Exception as e:
        print(f"Error in transcribe endpoint: {e}")
        return jsonify({"error": "Internal server error during transcription"}), 500

@app.route('/get_latest_response', methods=['GET'])
def get_latest_response():
    """Get the latest AI response"""
    if not current_session or not current_session.messages:
        return jsonify({"status": "no_messages"})
    
    # Get latest AI message
    latest_messages = [msg for msg in current_session.messages if msg["role"] == "assistant"]
    
    if not latest_messages:
        return jsonify({"status": "no_ai_response"})
    
    latest_response = latest_messages[-1]
    
    response_data = {
        "text": latest_response["content"],
        "language": latest_response.get("language", "en"),
        "timestamp": latest_response["timestamp"],
        "has_audio": "audio_file" in latest_response
    }
    
    return jsonify(response_data)

@app.route('/get_audio/<session_id>/<int:message_index>')
def get_audio(session_id, message_index):
    if not current_session or current_session.id != session_id:
        return jsonify({"error": "Session not found"}), 404
    
    try:
        if message_index < 0 or message_index >= len(current_session.messages):
            return jsonify({"error": "Message not found"}), 404
        
        message = current_session.messages[message_index]
        if not message.get("audio_file") or not os.path.exists(message["audio_file"]):
            return jsonify({"error": "Audio file not found"}), 404
        
        return send_file(
            message["audio_file"],
            mimetype="audio/mpeg",
            as_attachment=True,
            download_name=f"response_{message_index}.mp3"
        )
        
    except Exception as e:
        print(f"Error in get_audio endpoint: {e}")
        return jsonify({"error": "Internal server error while retrieving audio"}), 500

@app.route('/conversation_history')
def conversation_history():
    """Get current conversation history"""
    if not current_session:
        return jsonify({"messages": []})
    
    # Return messages without audio file paths (for security)
    messages = []
    for i, msg in enumerate(current_session.messages):
        message_data = {
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": msg["timestamp"],
            "language": msg.get("language", "auto"),
            "index": i,
            "has_audio": "audio_file" in msg
        }
        messages.append(message_data)
    
    return jsonify({
        "session_id": current_session.id,
        "messages": messages,
        "current_input": current_session.current_input
    })

@app.route('/stop_listening', methods=['POST'])
def stop_listening():
    """Manually stop listening and process accumulated input"""
    global silence_timer
    
    if silence_timer:
        silence_timer.cancel()
    
    # Process immediately
    handle_silence_timeout()
    
    return jsonify({"status": "processing_complete"})

@app.route('/process_text', methods=['POST'])
def process_text():
    global current_session
    
    if not current_session:
        return jsonify({"error": "No active session"}), 400
    
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        text = data['text'].strip()
        if not text:
            return jsonify({"error": "Empty text provided"}), 400
        
        # Add user message to session
        user_msg = current_session.add_message("user", text)
        
        # Get conversation context
        context = [{"role": msg["role"], "content": msg["content"]} 
                  for msg in current_session.messages[-10:]]
        
        # Process with AI
        ai_result = process_with_groq(text, context)
        
        if not ai_result or "response" not in ai_result:
            return jsonify({"error": "Failed to get AI response"}), 500
        
        # Add AI response to session
        ai_msg = current_session.add_message("assistant", ai_result["response"], ai_result["language"])
        
        # Generate TTS
        voice = "nova" if ai_result["language"] == "en" else "alloy"
        audio_file = generate_speech(ai_result["response"], ai_result["language"], voice)
        
        if audio_file:
            ai_msg["audio_file"] = audio_file
        
        return jsonify({
            "status": "success",
            "response": ai_result["response"],
            "language": ai_result["language"],
            "has_audio": bool(audio_file)
        })
        
    except Exception as e:
        print(f"Error in process_text endpoint: {e}")
        return jsonify({"error": "Internal server error during text processing"}), 500

@app.route('/clear_session', methods=['POST'])
def clear_session():
    global current_session
    
    if not current_session:
        return jsonify({"error": "No active session"}), 400
    
    try:
        session_id = current_session.id
        cleanup_session(session_id)
        current_session = None
        
        return jsonify({
            "status": "success",
            "message": "Session cleared successfully"
        })
    except Exception as e:
        print(f"Error clearing session: {e}")
        return jsonify({"error": "Failed to clear session"}), 500

@app.route('/set_silence_threshold', methods=['POST'])
def set_silence_threshold():
    """Set silence detection threshold"""
    data = request.get_json()
    threshold = data.get('threshold', 2.0)
    
    if current_session:
        current_session.silence_threshold = max(0.5, min(10.0, threshold))  # Clamp between 0.5-10 seconds
        
        return jsonify({
            "status": "success",
            "threshold": current_session.silence_threshold
        })
    
    return jsonify({"error": "No active session"}), 400

# Add session cleanup thread
session_cleanup_thread = threading.Thread(
    target=lambda: cleanup_expired_sessions(),
    daemon=True
)
session_cleanup_thread.start()

# Add shutdown handler
def cleanup_on_shutdown():
    """Clean up all resources when the server shuts down"""
    try:
        # Clean up all sessions
        for session_id in list(SESSIONS.keys()):
            cleanup_session(session_id)
        
        # Clean up temporary files
        cleanup_temp_files()
        
        # Shutdown thread pool
        executor.shutdown(wait=False)
        
    except Exception as e:
        print(f"Error during shutdown cleanup: {e}")

# Register shutdown handler
atexit.register(cleanup_on_shutdown)

if __name__ == '__main__':
    try:
        print("🎤 AI Voice Conversational Backend Starting...")
        print("🔊 Features: Speech-to-Text, AI Processing, Text-to-Speech")
        print("🌍 Languages: Hindi + English")
        print("⚡ Real-time conversation with silence detection")
        print("🚀 Server running on http://localhost:5000")
        
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        cleanup_temp_files()
    except Exception as e:
        print(f"❌ Server error: {e}")
        cleanup_temp_files()