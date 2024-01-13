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
import base64
from PIL import ImageFont, ImageDraw, Image

app = FastAPI()

urls = ['translate.googleapis.com', 'translate.google.com', 'translate.google.com.ar', 'translate.google.com.br',
        'translate.google.com']

user = 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:83.0) Gecko/20100101 Firefox/83.0'

translator = Translator(service_urls=urls, user_agent=user, raise_exception=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    "zh-CN": "chi_sim",
    "zh-TW": "chi_tra",
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


def detect_language(text: str):
    return translator.detect(text).lang


def translate_text(text: str, src: str = 'auto', dest: str = 'en'):
    return translator.translate(text=text, src=src, dest=dest).text


def get_tesseract_language(google_translate_code):
    return language_mapping.get(google_translate_code)


async def convert_bytes_in_image(image: UploadFile = File(...)):
    content = await image.read()
    nparr = np.frombuffer(content, np.uint8)
    return cv.imdecode(nparr, cv.IMREAD_COLOR)


async def convert_image_in_bytes(image: Mat):
    _, img_encoded = cv.imencode(".png", image)
    img_base64 = base64.b64encode(img_encoded).decode("utf-8")
    return img_base64


def transform_rgb_to_binary(img: Mat):
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gaussian_blur = cv.GaussianBlur(gray_image, (7, 7), 0)
    _, binary = cv.threshold(gaussian_blur, 200, 255, cv.THRESH_BINARY)
    binary = cv.bitwise_not(binary)
    return binary


async def extract_text_from_image(src: str, content: Mat = None):
    binary = transform_rgb_to_binary(content)

    extractedText = pt.image_to_string(binary, lang=src).strip()
    textPositions = pt.image_to_boxes(binary, lang=src)

    return extractedText, textPositions


def translate_and_replace_text_from_image(src: str, dest: str, content: Mat = None):
    if src == 'auto':
        src = 'en'

    ocr_src = get_tesseract_language(src)

    binary = transform_rgb_to_binary(content)
    textData = pt.image_to_data(binary, lang=ocr_src)

    imgReplaced = content.copy()

    linhas = textData.splitlines()

    for x, linha in enumerate(linhas):
        if x != 0:
            linha = linha.split()

            if len(linha) > 11:
                x, y, w, h = map(int, linha[6:10])

                word = translate_text(str(linha[11]), src=src, dest=dest)

                cv.rectangle(imgReplaced, (x, y), (w + x, h + y), (255, 255, 255), -1)

                imagem_pil = Image.fromarray(imgReplaced)

                draw = ImageDraw.Draw(imagem_pil)

                font = ImageFont.truetype("arial.ttf", int(h * 1.2), encoding='utf-8')

                draw.text((x, y), word, fill=(0, 0, 0), font=font, spacing=1, align='center')

                imgReplaced = cv.cvtColor(np.array(imagem_pil), cv.COLOR_RGB2BGR)

    return imgReplaced


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
async def api_extract_text_from_image(src: str = Body(...), image: UploadFile = File(...)):
    try:
        img = await convert_bytes_in_image(image=image)

        extractedText, textPositions = await extract_text_from_image(src=src, content=img)

        return {"extractedText": extractedText, "textPositions": textPositions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao extrair texto da imagem: {str(e)}")


@app.post("/api/translate-and-replace-text-from-image")
async def api_translate_and_replace_text_from_image(src: str = Body(...), dest: str = Body(...),
                                                    image: UploadFile = File(...)):
    try:
        img = await convert_bytes_in_image(image=image)

        imgReplaced = await translate_and_replace_text_from_image(src=src, dest=dest, content=img)
        imgConverted = convert_image_in_bytes(imgReplaced)

        return {"imgReplaced": imgConverted}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Erro ao extrair texto da imagem: {str(e)}")


@app.post("/api/extract-texts-from-images")
async def api_extract_texts_from_images(src: str = Body(...), images: List[UploadFile] = File(...)):
    try:
        extractedTexts = []
        extractedTextsPositions = []
        src = get_tesseract_language(src)

        for image in images:
            img = await convert_bytes_in_image(image=image)

            extractedText, textPositions = await extract_text_from_image(src=src, content=img)

            extractedTexts.append(extractedText)
            extractedTextsPositions.append(textPositions)
        return {"extractedTexts": extractedTexts, "extractedTextsPositions": extractedTextsPositions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao extrair texto das imagens: {str(e)}")


@app.post("/api/extract-text-from-video")
async def api_extract_text_from_video(src: str = Body(None), video: UploadFile = File(...)):
    try:
        video_path = "temp_video.mp4"
        with tempfile.NamedTemporaryFile(delete=False, dir=tempfile.gettempdir()) as video_file:
            video_file.write(video.file.read())

        clip = VideoFileClip(video_path)
        frames = [frame for frame in clip.iter_frames()]

        extractedTexts = []
        extractedTextsPositions = []
        for frame in frames:
            extractedText, textPositions = await extract_text_from_image(src=src, content=frame)

            extractedTexts.append(extractedText)
            extractedTextsPositions.append(textPositions)

        os.remove(video_path)

        return {"extractedTexts": extractedTexts, "extractedTextsPositions": extractedTextsPositions}
    except Exception as e:
        return {"error": f"Erro ao extrair texto do vídeo {str(e)}"}