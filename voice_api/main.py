# voice_api/main.py

import os
import sys
import tempfile
import subprocess
import logging
import io
import numpy as np
import scipy.io.wavfile
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
import whisper_timestamped as whisper
from typing import Dict, Any

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🖥️ Device: {device}")

# --- Global Variables for Models ---
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

# --- Dependency Setup ---
def setup_dependencies():
    """Download YarnGPT and WavTokenizer files automatically."""
    logger.info("🔍 Setting up dependencies...")

    # Clone YarnGPT if not exists
    if not os.path.exists('./yarngpt'):
        logger.info("📦 Cloning YarnGPT repository...")
        subprocess.run(['git', 'clone', 'https://github.com/saheedniyi02/yarngpt.git'], check=True)
        logger.info("✅ YarnGPT cloned")

    # Download config YAML
    if not os.path.exists('./wavtokenizer_config.yaml'):
        logger.info("📄 Downloading WavTokenizer config...")
        subprocess.run([
            'wget', '-q',
            'https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/resolve/main/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml',
            '-O', 'wavtokenizer_config.yaml'
        ], check=True)
        logger.info("✅ Config downloaded")

    # Download the 1.7GB model checkpoint
    if not os.path.exists('./wavtokenizer_model.ckpt'):
        logger.info("🔽 Downloading WavTokenizer model (1.7GB)...")
        subprocess.run([
            'gdown',
            '1-ASeEkrn4HY49yZWHTASgfGFNXdVnLTt',
            '-O', 'wavtokenizer_model.ckpt'
        ], check=True)
        logger.info("✅ Model downloaded")

# Add YarnGPT path to sys.path
if os.path.exists('./yarngpt'):
    sys.path.append('./yarngpt')

try:
    from audiotokenizer import AudioTokenizerV2
except ImportError:
    logger.warning("audiotokenizer import failed. Run setup_dependencies() first.")

# --- Lazy-load Models ---
@torch.no_grad()
def load_models():
    global whisper_model, yarngpt2_model, audio_tokenizer, yarngpt2_tokenizer

    if whisper_model is None:
        logger.info("🤖 Loading Whisper Medium model...")
        whisper_model = whisper.load_model("medium", device=device)
        logger.info("✅ Whisper loaded")

    if yarngpt2_model is None or audio_tokenizer is None:
        logger.info("🤖 Loading YarnGPT2 model...")
        YARNGPT_HF_MODEL = "saheedniyi/YarnGPT2"

        yarngpt2_tokenizer = AutoTokenizer.from_pretrained(YARNGPT_HF_MODEL)
        audio_tokenizer = AudioTokenizerV2(
            YARNGPT_HF_MODEL,
            "wavtokenizer_model.ckpt",
            "wavtokenizer_config.yaml"
        )
        yarngpt2_model = AutoModelForCausalLM.from_pretrained(
            YARNGPT_HF_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device)
        logger.info("✅ YarnGPT2 loaded")

# --- FastAPI App ---
app = FastAPI(
    title="AI HealthMate Voice API",
    description="STT/TTS Microservice for Nigerian Languages (Yoruba, Igbo, Hausa)",
    version="1.0.0",
    on_startup=[setup_dependencies, load_models]
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Core Functions ---
@torch.no_grad()
def transcribe_audio(audio_path: str):
    if whisper_model is None:
        raise RuntimeError("Whisper model not loaded")
    result = whisper_model.transcribe(audio_path, fp16=(device.type == 'cuda'))
    lang_code = result.get('language', 'en')
    return {
        "text": result['text'].strip(),
        "language_code": lang_code,
        "language_name": LANGUAGE_CODES.get(lang_code, 'english')
    }

@torch.no_grad()
def generate_speech_yarngpt2(text: str, language_name: str):
    if yarngpt2_model is None or audio_tokenizer is None:
        raise RuntimeError("YarnGPT2 models not loaded")
    speaker_name = YARNGPT_SPEAKERS.get(language_name.lower(), 'idera')
    prompt = audio_tokenizer.create_prompt(text, language_name.lower(), speaker_name)
    input_ids = audio_tokenizer.tokenize_prompt(prompt).to(device)
    output_codes = yarngpt2_model.generate(
        input_ids=input_ids,
        max_length=1024,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )
    codes = audio_tokenizer.get_codes(output_codes)
    audio_array = audio_tokenizer.get_audio(codes)
    if torch.is_tensor(audio_array):
        audio_array = audio_array.cpu().numpy()
    audio_array = np.squeeze(audio_array)
    return (audio_array * 32767).astype(np.int16)

# --- Endpoints ---
@app.post("/stt")
async def stt_endpoint(audio: UploadFile = File(...)):
    if not audio.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be audio.")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        audio_path = tmp.name
        await audio.close()
    try:
        result = transcribe_audio(audio_path)
        return {
            "success": True,
            "userText": result["text"],
            "detectedLanguage": result["language_name"],
            "languageCode": result["language_code"]
        }
    finally:
        os.unlink(audio_path)

@app.post("/tts")
async def tts_endpoint(
    text: str = Form(...),
    language: str = Form(...)
):
    if language.lower() not in YARNGPT_SPEAKERS:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
    audio_int16 = generate_speech_yarngpt2(text, language)
    wav_buffer = io.BytesIO()
    scipy.io.wavfile.write(wav_buffer, YARNGPT_RATE, audio_int16)
    wav_buffer.seek(0)
    return StreamingResponse(
        wav_buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=response.wav"}
    )
