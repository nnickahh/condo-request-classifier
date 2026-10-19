"""
Multi-language translation and audio speech-to-text transcription service
for the condominium management resident bot.

Features:
- Robust language detection (Chinese, Malay, Tamil, Japanese, Korean, Spanish, French, German, etc.)
- Transparent translation to English for internal ML triage, work order logging, and staff dispatch
- Resident response translation back to the resident's preferred/native language
- Audio / Voice message transcription to English text using SpeechRecognition and soundfile
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, NamedTuple

logger = logging.getLogger("condo_jev.translator")

# Supported language codes and friendly names
SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "zh": "Chinese (中文)",
    "ms": "Malay (Bahasa Melayu)",
    "ta": "Tamil (தமிழ்)",
    "ja": "Japanese (日本語)",
    "ko": "Korean (한국어)",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "de": "German (Deutsch)",
    "id": "Indonesian (Bahasa Indonesia)",
    "vi": "Vietnamese (Tiếng Việt)",
    "th": "Thai (ไทย)",
    "hi": "Hindi (हिन्दी)",
    "ar": "Arabic (العربية)",
    "ru": "Russian (Русский)",
}


class TranslationResult(NamedTuple):
    original_text: str
    translated_text: str
    detected_lang: str
    detected_lang_name: str
    is_translated: bool


def detect_language(text: str) -> tuple[str, str]:
    """
    Detect the likely language of the given text using Unicode script analysis
    and frequent localized keyword heuristics.
    Returns (lang_code, lang_name).
    """
    if not text or not text.strip():
        return "en", "English"

    s = text.strip()

    # CJK Unified Ideographs (Chinese)
    if re.search(r"[\u4e00-\u9fff]", s):
        return "zh", "Chinese (中文)"

    # Japanese Hiragana / Katakana
    if re.search(r"[\u3040-\u30ff]", s):
        return "ja", "Japanese (日本語)"

    # Korean Hangul
    if re.search(r"[\uac00-\ud7af]", s):
        return "ko", "Korean (한국어)"

    # Tamil script
    if re.search(r"[\u0b80-\u0bff]", s):
        return "ta", "Tamil (தமிழ்)"

    # Devanagari (Hindi)
    if re.search(r"[\u0900-\u097f]", s):
        return "hi", "Hindi (हिन्दी)"

    # Arabic script
    if re.search(r"[\u0600-\u06ff]", s):
        return "ar", "Arabic (العربية)"

    # Cyrillic (Russian)
    if re.search(r"[\u0400-\u04ff]", s):
        return "ru", "Russian (Русский)"

    # Thai script
    if re.search(r"[\u0e00-\u0e7f]", s):
        return "th", "Thai (ไทย)"

    # Vietnamese special diacritics
    if re.search(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", s, re.I):
        return "vi", "Vietnamese (Tiếng Việt)"

    # Malay / Indonesian common vocabulary
    if re.search(r"\b(bocor|rosak|pintu|tangga|tolong|lif|lampu|tandas|sampah|tingkat|kolam|pembinaan|anjing|bising|tersekat|kebakaran|kecemasan|pegawai|selamat|pagi|petang|malam|terima\s+kasih|blok|kolam\s+renang|paip)\b", s, re.I):
        return "ms", "Malay (Bahasa Melayu)"

    # Spanish common vocabulary
    if re.search(r"\b(fuego|humo|fuga|ruido|gracias|hola|ascensor|puerta|agua|ayuda|por\s+favor|piscina|aparcamiento|habitacion|edificio)\b", s, re.I):
        return "es", "Spanish (Español)"

    # French common vocabulary
    if re.search(r"\b(bonjour|merci|fuite|bruit|eau|ascenseur|incendie|secours|porte|voisin|piscine|stationnement|chambre)\b", s, re.I):
        return "fr", "French (Français)"

    # German common vocabulary
    if re.search(r"\b(hallo|danke|wasser|aufzug|lärm|feuer|bitte|wohnung|notfall|tür|pool|treppe)\b", s, re.I):
        return "de", "German (Deutsch)"

    return "en", "English"


def translate_text(text: str, target_lang: str = "en", source_lang: str = "auto") -> TranslationResult:
    """
    Translate text between languages using MyMemory Translation API with graceful fallback.
    Returns TranslationResult containing translated text, detected language, and metadata.
    """
    clean = text.strip()
    if not clean:
        return TranslationResult(clean, clean, "en", "English", False)

    detected_code, detected_name = detect_language(clean)

    # Normalize language codes
    tl = target_lang.lower().strip()
    sl = source_lang.lower().strip()

    if sl == "auto":
        sl = detected_code

    if sl == "zh":
        sl = "zh-CN"
    if tl == "zh":
        tl = "zh-CN"

    # If already in the target language
    if sl == tl or (detected_code == "en" and tl == "en"):
        return TranslationResult(
            original_text=clean,
            translated_text=clean,
            detected_lang=detected_code,
            detected_lang_name=detected_name,
            is_translated=False,
        )

    # Query external translation API
    try:
        langpair = f"{sl}|{tl}"
        url = f"https://api.mymemory.translated.net/get?q={urllib.parse.quote(clean)}&langpair={langpair}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CondoManagementBot/1.0"},
        )
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            translated = data.get("responseData", {}).get("translatedText")

            # Check for API error strings returned inside translatedText
            if translated and not translated.startswith("QUERY LENGTH LIMIT") and not translated.startswith("MYMEMORY WARNING"):
                # Clean up any HTML entities like &#39;
                import html
                translated = html.unescape(translated).strip()
                return TranslationResult(
                    original_text=clean,
                    translated_text=translated,
                    detected_lang=detected_code,
                    detected_lang_name=detected_name,
                    is_translated=True,
                )
    except Exception as exc:
        logger.warning(f"Translation API request failed ({exc}); falling back to original text.")

    # Fallback: return original text
    return TranslationResult(
        original_text=clean,
        translated_text=clean,
        detected_lang=detected_code,
        detected_lang_name=detected_name,
        is_translated=False,
    )


def transcribe_audio_to_english(audio_path: str | Path, language: str | None = None) -> str | None:
    """
    Transcribe a recorded voice note or audio file to English text.
    Handles Telegram Opus OGG, MP3, WAV, etc., by converting to PCM WAV in memory
    and querying SpeechRecognition.
    Returns transcribed string, or None if audio cannot be transcribed.
    """
    p = Path(audio_path)
    if not p.is_file():
        logger.warning(f"Audio file does not exist: {audio_path}")
        return None

    try:
        import soundfile as sf
        import speech_recognition as sr

        # Read audio file using soundfile
        audio_data, sample_rate = sf.read(str(p), dtype="float32")

        # Convert to mono if multi-channel
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Write to in-memory 16-bit PCM WAV at 16kHz
        wav_io = io.BytesIO()
        sf.write(wav_io, audio_data, sample_rate, format="WAV", subtype="PCM_16")
        wav_io.seek(0)

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            recorded = recognizer.record(source)

        # First attempt: transcribe with specified language or default en-SG/en-US
        speech_lang = language or "en-SG"
        try:
            transcript = recognizer.recognize_google(recorded, language=speech_lang)
        except sr.UnknownValueError:
            # Try secondary general English (en-US)
            try:
                transcript = recognizer.recognize_google(recorded, language="en-US")
            except sr.UnknownValueError:
                # Try multilingual autodetect fallback (e.g. Chinese zh-CN or Malay ms-MY)
                for alt_lang in ("zh-CN", "ms-MY"):
                    try:
                        transcript = recognizer.recognize_google(recorded, language=alt_lang)
                        if transcript:
                            break
                    except Exception:
                        pass
                else:
                    return None

        if transcript and transcript.strip():
            clean_transcript = transcript.strip()
            # If transcription was in a foreign language, translate it to English!
            det_lang, _ = detect_language(clean_transcript)
            if det_lang != "en":
                res = translate_text(clean_transcript, target_lang="en", source_lang=det_lang)
                if res.is_translated:
                    return res.translated_text
            return clean_transcript

        return None
    except Exception as exc:
        logger.warning(f"Audio transcription error for {audio_path}: {exc}")
        return None
