from cv2 import Mat
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from moviepy.editor import VideoFileClip
import cv2 as cv
from pytesseract import pytesseract as pt, Output
from googletrans import Translator, LANGUAGES
import numpy as np
import tempfile
from typing import List
from typing import Union
import requests
from PIL import ImageFont, ImageDraw, Image
from unidecode import unidecode
from textwrap import wrap
from io import BytesIO
import os

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

language_mapping = {}


def read_language_mapping(filename="language_mapping.txt"):
    with open(filename, "r") as file:
        for line in file:
            key, value = line.strip().split(":")
            language_mapping[key] = value


read_language_mapping()


def detect_language(text: str):
    return translator.detect(text).lang


def translate_text(text: str, src: str = 'auto', dest: str = 'en'):
    return translator.translate(text=text, src=src, dest=dest).text


def get_tesseract_language(google_translate_code):
    return language_mapping.get(google_translate_code)


async def convert_bytes_in_image(image: UploadFile):
    content = await image.read()
    np_arr = np.frombuffer(content, np.uint8)
    return cv.imdecode(np_arr, cv.IMREAD_COLOR)


async def convert_image_in_bytes(img: Image) -> bytes:
    img_byte_array = BytesIO()
    img.save(img_byte_array)
    return img_byte_array.getvalue()


def transform_rgb_to_binary(img: Mat):
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray_image, 200, 255, cv.THRESH_BINARY)
    binary = cv.bitwise_not(binary)
    return binary


async def extract_text_from_image(src: str, content: Mat = None):
    binary = transform_rgb_to_binary(content)
    text_data = pt.image_to_data(binary, lang=src)
    return text_data


def get_lines(paragraph):
    lines = []
    current_line = {'x': None, 'y': None, 'w': None, 'h': None, 'text': '', 'length': 0}

    for word in paragraph:
        x, y, w, h, text = word

        if current_line['x'] is None:
            current_line['x'] = x
            current_line['y'] = y
            current_line['w'] = w
            current_line['h'] = h

        current_line['text'] = (current_line['text'] + ' ' + text.strip()).strip()
        current_line['length'] = len(current_line['text'])

        current_line['w'] = word[0] + word[2]

        if text.endswith('\n'):
            current_line['h'] = int(sum(item[3] for item in paragraph) / len(paragraph))

            lines.append(current_line)
            current_line = {'x': None, 'y': None, 'w': None, 'h': None, 'text': '', 'length': 0}

    if current_line['text']:
        current_line['h'] = int(sum(item[3] for item in paragraph) / len(paragraph))
        lines.append(current_line)

    return lines


def get_paragraphs(text_data):
    paragraphs = []
    current_paragraph = []
    num_items = len(text_data['text'])
    last_block_num = 0
    last_line_num = 0

    print(text_data)

    for i in range(num_items):
        if int(text_data['conf'][i]) > 50:
            x, y, w, h, block_num, line_num = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i], text_data['block_num'][i], text_data['line_num'][i]

            if block_num != 0 and last_line_num != 0:
                if line_num != last_line_num:
                    current_paragraph[-1][4] = current_paragraph[-1][4] + '\n'

                if last_block_num != block_num:
                    paragraphs.append(current_paragraph)
                    current_paragraph = []

            current_paragraph.append([x, y, w, h, text_data['text'][i].strip()])
            last_block_num = block_num
            last_line_num = line_num

    if current_paragraph:
        paragraphs.append(current_paragraph)

    for i in range(len(paragraphs)):
        lines = get_lines(paragraphs[i])
        x = min(line['x'] for line in lines)
        y = min(line['y'] for line in lines)
        w = max(line['x'] + line['w'] for line in lines) - x
        h = max(line['y'] + line['h'] for line in lines) - y
        text = ' '.join(line['text'] for line in lines)
        length = max(line['length'] for line in lines) + 2
        paragraphs[i] = {'x': x, 'y': y, 'w': w, 'h': h, 'text': text, 'lines': lines, 'length': length}

    return paragraphs


def translate_and_replace_text_from_image(src: str, dest: str, content: Mat = None):
    if src == 'auto':
        src = 'en'

    ocr_src = get_tesseract_language(src)

    binary = transform_rgb_to_binary(content)

    text_data = pt.image_to_data(binary, lang=ocr_src, output_type=Output.DICT)

    img_replaced = content.copy()

    img_pil = Image.fromarray(img_replaced)
    draw = ImageDraw.Draw(img_pil)

    paragraphs = get_paragraphs(text_data)

    for paragraph in paragraphs:
        x, y, w, h, text, lines, length = paragraph['x'], paragraph['y'], paragraph['w'], paragraph['h'], paragraph['text'], paragraph['lines'], paragraph['length']

        translated_text = translate_text(text, src=src, dest=dest)

        translated_lines = wrap(text=translated_text, width=length)

        for i, line in enumerate(lines):
            line_x, line_y, line_w, line_h, original_line_text = line['x'], line['y'], line['w'], line['h'], line['text']

            translated_line = translated_lines[i]

            contains_special_character = unidecode(original_line_text) != original_line_text or unidecode(translated_line) != translated_line

            if contains_special_character:
                aditional_size = int(line_h / 3)
            else:
                aditional_size = 0

            draw.rectangle((line_x, line_y - aditional_size, line_w, line_h + line_y), fill='white')

            font_size = int(line_h * 1.2)
            font = ImageFont.truetype("arial.ttf", font_size, encoding='utf-8')

            if i < len(translated_lines):
                draw.text((line_x, line_y), translated_line, fill=(0, 0, 0), font=font)

    imgReplaced = cv.cvtColor(np.array(img_pil), cv.COLOR_RGB2BGR)

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

        text_data = await extract_text_from_image(src=src, content=img)

        return {"text_data": text_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao extrair texto da imagem: {str(e)}")


@app.post("/api/translate-and-replace-text-from-image")
async def api_translate_and_replace_text_from_image(
    src: str = Body(...),
    dest: str = Body(...),
    image: Union[UploadFile, str] = File(...)
):
    try:
        if isinstance(image, str):
            print('é')
            response = requests.get(image)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        else:
            img = await convert_bytes_in_image(image=image)

        imgReplaced = translate_and_replace_text_from_image(src=src, dest=dest, content=img)
        imgConverted = await convert_image_in_bytes(imgReplaced)

        return StreamingResponse(BytesIO(imgConverted), media_type="image/jpeg")
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Erro ao fazer a requisição HTTP: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao carregar imagem: {str(e)}")


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