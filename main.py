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
from pathlib import Path

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


async def convert_image_in_bytes(img: Mat, etx: str) -> bytes:
    _, img_encoded = cv.imencode(ext=etx, img=img)
    return img_encoded.tobytes()


async def get_image_data_from_bytes(image: UploadFile):
    if isinstance(image, str):
        response = requests.get(image)
        response.raise_for_status()
        image = response.content

    img = await convert_bytes_in_image(image=image)
    etx = Path(image.filename).suffix.lower()

    return img, etx


async def transform_rgb_to_binary(img):
    image_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image_binary = cv.adaptiveThreshold(image_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 4)
    image_closing = cv.morphologyEx(image_binary, cv.MORPH_CLOSE, (7, 7), iterations=1)
    return image_closing


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
            current_line['h'] = int(np.sum(np.array([item[3] for item in paragraph])) / len(paragraph))

            lines.append(current_line)
            current_line = {'x': None, 'y': None, 'w': None, 'h': None, 'text': '', 'length': 0}

    if current_line['text']:
        current_line['h'] = int(np.sum(np.array([item[3] for item in paragraph])) / len(paragraph))
        lines.append(current_line)

    return lines


def is_compatible_with_paragraphs(x, y, w, h, paragraphs):
    for paragraph in paragraphs:
        para_x, para_y, para_w, para_h, _ = paragraph[0]
        if para_x + para_w + w > x and para_y + para_h + h > y:
            return paragraph
    return False


def get_paragraphs(text_data):
    paragraphs = []
    current_paragraph = []
    num_items = len(text_data['text'])
    last_block_num = 0
    current_line = {'x': 0, 'y': 0, 'w': 0, 'h': 0, 'num': 0}
    block_nums = set()

    for i in range(num_items):
        x, y, w, h, block_num, line_num, text = text_data['left'][i], text_data['top'][i], text_data['width'][i], \
        text_data['height'][i], text_data['block_num'][i], text_data['line_num'][i], text_data['text'][i]

        if int(text_data['conf'][i]) > 36 and text.strip() != '':
            if block_num != 0 and current_line['num'] != 0:
                if last_block_num != block_num:
                    if is_compatible_with_paragraphs(x, y, w, h, paragraphs):
                        current_paragraph = next((paragraph for paragraph in paragraphs if paragraph[0][4] == block_num), [])

                    else:
                        paragraphs.append(current_paragraph)
                        current_paragraph = []
                        block_nums.add(block_num)

                if line_num != current_line['num'] or (((y - current_line['y'] > current_line['h']) or (
                        x - current_line['x'] > current_line['w'] * 2))):
                    if current_paragraph:
                        current_paragraph[-1][4] = current_paragraph[-1][4] + '\n'
                    current_line = {'x': 0, 'y': 0, 'w': 0, 'h': 0, 'num': 0}

            current_paragraph.append([x, y, w, h, text_data['text'][i].strip()])
            last_block_num = block_num
            current_line['num'] = line_num
            current_line['x'] = x
            current_line['y'] = y
            current_line['w'] = w
            current_line['h'] = h

    if current_paragraph:
        paragraphs.append(current_paragraph)

    for i in range(len(paragraphs)):
        lines = get_lines(paragraphs[i])
        if lines:
            x = min(line['x'] for line in lines)
            y = min(line['y'] for line in lines)
            w = max(line['x'] + line['w'] for line in lines) - x
            h = max(line['y'] + line['h'] for line in lines) - y
            text = ' '.join(line['text'] for line in lines)
            length = max(line['length'] for line in lines) + 2
            paragraphs[i] = {'x': x, 'y': y, 'w': w, 'h': h, 'text': text, 'lines': lines, 'length': length}

    return paragraphs


def draw_rectangles(draw, lines):
    for line in lines:
        line_x, line_y, line_w, line_h, original_line_text = line['x'], line['y'], line['w'], line['h'], line['text']

        contains_special_character = unidecode(original_line_text) != original_line_text

        if contains_special_character:
            additional_size = int(line_h / 3)
        else:
            additional_size = 0

        draw.rectangle((line_x, line_y - additional_size, line_w, line_h + line_y + additional_size), fill='white')


def translate_and_draw(draw, lines, translated_lines):
    for i, line in enumerate(lines):
        x, y, h = line['x'], line['y'], line['h']

        if i < len(translated_lines):
            font = ImageFont.truetype("impact.ttf", int(h * 1.25), encoding='unic')
            translated_line = translated_lines[i]
            draw.text((x, y), translated_line, fill=(0, 0, 0), font=font)


async def translate_and_replace_text_from_image(src: str, dest: str, content: Mat = None):
    if src == 'auto':
        src = 'en'

    ocr_src = get_tesseract_language(src)

    binary = await transform_rgb_to_binary(content)

    text_data = pt.image_to_data(binary, lang=ocr_src, output_type=Output.DICT)

    img_replaced = content.copy()

    img_pil = Image.fromarray(img_replaced)
    draw = ImageDraw.Draw(img_pil)

    paragraphs = get_paragraphs(text_data)

    for i, paragraph in enumerate(paragraphs):
        if isinstance(paragraph, list):
            continue

        x, y, w, h, text = paragraph['x'], paragraph['y'], paragraph['w'], paragraph['h'], paragraph['text']

        if text.strip() == '':
            continue

        lines, length = paragraph['lines'], paragraph['length']

        translated_text = translate_text(text, src=src, dest=dest)
        translated_lines = wrap(text=translated_text, width=length)

        draw_rectangles(draw, lines)
        translate_and_draw(draw, lines, translated_lines)

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
        img, etx = await get_image_data_from_bytes(image=image)

        imgReplaced = await translate_and_replace_text_from_image(src=src, dest=dest, content=img)
        imgConverted = await convert_image_in_bytes(etx=etx, img=imgReplaced)

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