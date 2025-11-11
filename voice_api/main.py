import os
import sys
import tempfile
import subprocess
import logging
import io
import numpy as np
import scipy.io.wavfile
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
import whisper_timestamped as whisper
from typing import Dict, Any

# --- Logger Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Global Configuration & Model Variables ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_model: Any = None
yarngpt2_model: Any = None
audio_tokenizer: Any = None
yarngpt2_tokenizer: Any = None

LANGUAGE_CODES: Dict[str, str] = {'en': 'english', 'yo': 'yoruba', 'ig': 'igbo', 'ha': 'hausa'}
YARNGPT_SPEAKERS: Dict[str, str] = {
    'english': 'idera',
    'yoruba': 'yoruba_male2',
    'igbo': 'igbo_female2',
    'hausa': 'hausa_female1'
}
WHISPER_RATE = 16000
YARNGPT_RATE = 24000
YARNGPT_HF_MODEL = "saheedniyi/YarnGPT2"


# --- Setup Dependencies Function ---
def setup_dependencies():
    """Download YarnGPT and WavTokenizer files automatically."""
    logger.info("🔍 Setting up dependencies...")

    # 1. Clone YarnGPT if missing
    if not os.path.exists('./yarngpt'):
        logger.info("📦 Cloning YarnGPT...")
        subprocess.run(['git', 'clone', 'https://github.com/saheedniyi02/yarngpt.git'], check=True)

    # 2. Download WavTokenizer config if missing
    if not os.path.exists('./wavtokenizer_config.yaml'):
        logger.info("📄 Downloading WavTokenizer config...")
        subprocess.run([
            'wget', '-q',
            'https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/resolve/main/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml',
            '-O', 'wavtokenizer_config.yaml'
        ], check=True)

    # 3. Download WavTokenizer model if missing
    if not os.path.exists('./wavtokenizer_model.ckpt'):
        logger.info("🔽 Downloading WavTokenizer model (1.7GB)...")
        # Ensure gdown is installed (it was in your requirements.txt)
        subprocess.run(['gdown', '1-ASeEkrn4HY49yZWHTASgfGFNXdVnLTt', '-O', 'wavtokenizer_model.ckpt'], check=True)

# Add YarnGPT path to system path for importing
if os.path.exists('./yarngpt'):
    sys.path.append('./yarngpt')

try:
    # This import depends on the 'yarngpt' folder being in sys.path
    from audiotokenizer import AudioTokenizerV2
except ImportError:
    # This warning is expected to be displayed during initial setup
    logger.warning("audiotokenizer import failed. Run setup_dependencies() first.")


# --- Lazy-Load Model Functions ---

def ensure_whisper_loaded():
    global whisper_model
    if whisper_model is None:
        logger.info("🤖 Loading Whisper Medium...")
        # fp16=False for CPU or smaller CUDA memory, or (device.type == 'cuda') for optimal CUDA usage
        whisper_model = whisper.load_model("medium", device=device, fp16=(device.type == 'cuda'))
        logger.info("✅ Whisper loaded")

def ensure_yarngpt_loaded():
    global yarngpt2_model, yarngpt2_tokenizer, audio_tokenizer
    if yarngpt2_model is None or yarngpt2_tokenizer is None or audio_tokenizer is None:
        logger.info("🤖 Loading YarnGPT2 + AudioTokenizer...")
        
        # Check if the necessary files are available before initializing AudioTokenizer
        if not all(os.path.exists(f) for f in ["wavtokenizer_model.ckpt", "wavtokenizer_config.yaml"]):
             logger.error("WavTokenizer files are missing. Cannot load YarnGPT. Ensure setup_dependencies() ran successfully.")
             raise RuntimeError("Missing WavTokenizer files.")

        yarngpt2_tokenizer = AutoTokenizer.from_pretrained(YARNGPT_HF_MODEL)
        audio_tokenizer = AudioTokenizerV2(
            YARNGPT_HF_MODEL,
            "wavtokenizer_model.ckpt",
            "wavtokenizer_config.yaml"
        )
        yarngpt2_model = AutoModelForCausalLM.from_pretrained(
            YARNGPT_HF_MODEL,
            # Use float16 on GPU for memory/speed, otherwise use float32 on CPU
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32 
        ).to(device)
        logger.info("✅ YarnGPT2 loaded")


# --- Eager Model Loading (for external script/notebook use) ---

def load_models():
    """Eagerly loads all models. Used for initial setup or testing."""
    setup_dependencies() 
    ensure_whisper_loaded()
    ensure_yarngpt_loaded()
    logger.info("🎉 All models loaded successfully!")


# --- FastAPI Setup ---
app = FastAPI(
    title="AI HealthMate Voice API",
    description="STT/TTS Microservice for Nigerian Languages",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 🚀 Application Startup Event for Eager Loading ---

@app.on_event("startup")
def startup_event():
    """
    Called when the application starts. 
    This is the best place to eagerly load models for production.
    """
    logger.info("⚡ Running FastAPI startup events...")
    # Eagerly load all dependencies and models on startup
    load_models() 
    logger.info("✅ FastAPI startup complete. Server ready.")


# --- STT Endpoint ---
@app.post("/stt")
async def stt_endpoint(audio: UploadFile = File(...)):
    if not audio.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be audio.")
    
    # Model is guaranteed to be loaded by the startup event
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        audio_path = tmp.name
        
    try:
        # Pass fp16 setting based on device
        result = whisper_model.transcribe(audio_path, fp16=(device.type == 'cuda'))
        lang_code = result.get('language', 'en')
        return {
            "success": True,
            "userText": result['text'].strip(),
            "detectedLanguage": LANGUAGE_CODES.get(lang_code, 'english'),
            "languageCode": lang_code
        }
    finally:
        os.unlink(audio_path)

# --- TTS Endpoint ---
@app.post("/tts")
async def tts_endpoint(text: str = Form(...), language: str = Form(...)):
    lang_key = language.lower()
    if lang_key not in YARNGPT_SPEAKERS:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
    
    # Model is guaranteed to be loaded by the startup event
    speaker_name = YARNGPT_SPEAKERS.get(lang_key, 'idera')
    
    try:
        # 1. Generate prompt tokens
        prompt = audio_tokenizer.create_prompt(text, lang_key, speaker_name)
        input_ids = audio_tokenizer.tokenize_prompt(prompt).to(device)
        
        # 2. Generate audio codes
        output_codes = yarngpt2_model.generate(
            input_ids=input_ids,
            max_length=1024,
            do_sample=True,
            temperature=0.8,
            top_p=0.9
        )
        codes = audio_tokenizer.get_codes(output_codes)
        
        # 3. Decode codes to audio array
        audio_array = audio_tokenizer.get_audio(codes)
        
        # 4. Prepare WAV response
        if torch.is_tensor(audio_array):
            audio_array = audio_array.cpu().numpy()
            
        # Rescale and convert to 16-bit integer format
        audio_int16 = (np.squeeze(audio_array) * 32767).astype(np.int16)
        
        wav_buffer = io.BytesIO()
        scipy.io.wavfile.write(wav_buffer, YARNGPT_RATE, audio_int16)
        wav_buffer.seek(0)
        
        return StreamingResponse(
            wav_buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=response.wav"}
        )
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during TTS generation.")
