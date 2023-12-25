from cv2 import Mat
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from moviepy.editor import VideoFileClip
import cv2 as cv
from pytesseract import pytesseract as pt
from googletrans import Translator, LANGUAGES
import numpy as np
import tempfile
from typing import List
import os
import traceback

app = FastAPI()
urls = ['translate.googleapis.com', 'translate.google.com', 'translate.google.com.ar', 'translate.google.com.br',
        'translate.google.com']
user = 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:83.0) Gecko/20100101 Firefox/83.0'
translator = Translator(service_urls=urls, user_agent=user, raise_exception=False)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou use um domínio específico
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

language_mapping = {
    "af": "afr",
    "sq": "sqi",
    "am": "amh",
    "ar": "ara",
    "hy": "hye",
    "az": "aze",
    "eu": "eus",
    "be": "bel",
    "bn": "ben",
    "bs": "bos",
    "bg": "bul",
    "ca": "cat",
    "ceb": "ceb",
    "ny": "nya",
    "zh-CN": "chi_sim",  # Chinês simplificado
    "zh-TW": "chi_tra",  # Chinês tradicional
    "co": "cos",
    "hr": "hrv",
    "cs": "ces",
    "da": "dan",
    "nl": "nld",
    "en": "eng",
    "eo": "epo",
    "et": "est",
    "tl": "fil",
    "fi": "fin",
    "fr": "fra",
    "fy": "fry",
    "gl": "glg",
    "ka": "kat",
    "de": "deu",
    "el": "ell",
    "gu": "guj",
    "ht": "hat",
    "ha": "hau",
    "haw": "haw",
    "iw": "heb",
    "hi": "hin",
    "hmn": "hmn",
    "hu": "hun",
    "is": "isl",
    "ig": "ibo",
    "id": "ind",
    "ga": "gle",
    "it": "ita",
    "ja": "jpn",
    "jw": "jav",
    "kn": "kan",
    "kk": "kaz",
    "km": "khm",
    "rw": "kin",
    "ko": "kor",
    "ku": "kur",
    "ky": "kir",
    "lo": "lao",
    "la": "lat",
    "lv": "lav",
    "lt": "lit",
    "lb": "ltz",
    "mk": "mkd",
    "mg": "mlg",
    "ms": "msa",
    "ml": "mal",
    "mt": "mlt",
    "mi": "mri",
    "mr": "mar",
    "mn": "mon",
    "my": "mya",
    "ne": "nep",
    "no": "nor",
    "ps": "pus",
    "fa": "fas",
    "pl": "pol",
    "pt": "por",
    "pa": "pan",
    "ro": "ron",
    "ru": "rus",
    "sm": "smo",
    "gd": "gla",
    "sr": "srp",
    "st": "sot",
    "sn": "sna",
    "sd": "snd",
    "si": "sin",
    "sk": "slk",
    "sl": "slv",
    "so": "som",
    "es": "spa",
    "su": "sun",
    "sw": "swa",
    "sv": "swe",
    "tg": "tgk",
    "ta": "tam",
    "te": "tel",
    "th": "tha",
    "tr": "tur",
    "uk": "ukr",
    "ur": "urd",
    "uz": "uzb",
    "vi": "vie",
    "cy": "cym",
    "xh": "xho",
    "yi": "yid",
    "yo": "yor",
    "zu": "zul",
}


def get_tesseract_language(google_translate_code):
    return language_mapping.get(google_translate_code, google_translate_code)


def detect_language_or_default(text: str):
    try:
        detected_language = detect_language(text)
        return detected_language
    except Exception as e:
        print(f"Error detecting language: {e}")
        return 'en'  # Defina a linguagem padrão desejada aqui


def detect_language(text: str):
    return translator.detect(text).lang


def translate_text(text: str, src: str = 'auto', dest: str = 'en'):
    return translator.translate(text=text, src=src, dest=dest).text


async def convert_bytes_in_image(image: UploadFile = File(...)):
    content = await image.read()
    nparr = np.frombuffer(content, np.uint8)
    return cv.imdecode(nparr, cv.IMREAD_COLOR)

async def extract_text_from_image(lang: str = None, content: Mat = None):
    gray_image = cv.cvtColor(content, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray_image, 50, 150, cv.THRESH_BINARY)
    # kernel = np.ones((5, 5), np.uint8)
    # dilatation = cv.erode(binary, kernel, iterations=1)
    extractedText = pt.image_to_string(binary, lang=lang).strip()
    textPositions = pt.image_to_boxes(binary, lang=lang)

    return extractedText, textPositions


@app.get("/api/supported-languages")
async def api_supported_languages():
    try:
        return {"supportedLanguages": LANGUAGES}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao obter idiomas suportados: {str(e)}")


@app.post("/api/detect-language")
async def api_detect_language(text: str = Body(...)):
    try:
        detected_language = await detect_language(text)
        return {"detectedLanguage": detected_language}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao detectar idioma: {str(e)}")


@app.post("/api/translate-text")
async def api_translate_text(text: str = Body(...), src: str = Body(default='auto'), dest: str = Body(default='en')):
    try:
        if (src in LANGUAGES or src == 'auto') and dest in LANGUAGES:
            translated_text = translate_text(text=text, dest=dest, src=src)
            return {"translatedText": translated_text}
        else:
            raise HTTPException(status_code=400, detail="Linguagem não suportada.")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao traduzir texto: {str(e)}")


@app.post("/api/translate-multiple")
async def api_translate_multiple(texts: List[str] = Body(...), dest: str = Body(...), src: str = Body(...), ):
    try:
        translated_texts = [await translate_text(text=text, dest=dest, src=src) for text in texts]
        return {"translatedTexts": translated_texts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao traduzir textos: {str(e)}")


@app.post("/api/translate-multiple-for-multiple")
async def api_translate_multiples(texts: List[str] = Body(...), destLanguages: List[str] = Body(...)):
    try:
        translations = {lang: translate_text(text=text, dest=lang) for text, lang in zip(texts, destLanguages)}
        return {"translations": translations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao traduzir textos: {str(e)}")


@app.post("/api/extract-text-from-image")
async def api_extract_text_from_image(lang: str = Body(...), image: UploadFile = File(...)):
    try:
        img = await convert_bytes_in_image(image=image)

        if lang == 'auto':
            lang = detect_language_or_default(image.filename)

        lang = get_tesseract_language(lang)
        extractedText, textPositions = await extract_text_from_image(lang=lang, content=img)

        return {"extractedText": extractedText, "textPositions": textPositions}
    except Exception as e:
        print(f"Erro ao extrair texto da imagem: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao extrair texto da imagem: {str(e)}")


@app.post("/api/extract-texts-from-images")
async def api_extract_texts_from_images(lang: str = Body(...), images: List[UploadFile] = File(...)):
    try:
        extractedTexts = []
        extractedTextsPositions = []
        lang = get_tesseract_language(lang)

        for image in images:
            img = await convert_bytes_in_image(image=image)

            extractedText, textPositions = await extract_text_from_image(lang=lang, content=img)

            extractedTexts.append(extractedText)
            extractedTextsPositions.append(textPositions)
        return {"extractedTexts": extractedTexts, "extractedTextsPositions": extractedTextsPositions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao extrair texto das imagens: {str(e)}")


@app.post("/api/extract-text-from-video")
async def api_extract_text_from_video(lang: str = Body(None), video: UploadFile = File(...)):
    try:
        # Salvar o vídeo temporariamente
        video_path = "temp_video.mp4"
        with tempfile.NamedTemporaryFile(delete=False, dir=tempfile.gettempdir()) as video_file:
            video_file.write(video.file.read())

        # Extrair legendas do vídeo usando a biblioteca moviepy
        clip = VideoFileClip(video_path)
        frames = [frame for frame in clip.iter_frames()]

        extractedTexts = []
        extractedTextsPositions = []
        for frame in frames:
            extractedText, textPositions = await extract_text_from_image(lang=lang, content=frame)

            extractedTexts.append(extractedText)
            extractedTextsPositions.append(textPositions)

        # Remover o vídeo temporário após a extração
        os.remove(video_path)

        return {"extractedTexts": extractedTexts, "extractedTextsPositions": extractedTextsPositions}
    except Exception as e:
        return {"error": f"Erro ao extrair texto do vídeo {e}"}
