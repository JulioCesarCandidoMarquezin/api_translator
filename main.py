from textwrap import wrap

from time import time
import asyncio

import pytesseract
from unidecode import unidecode

import cv2 as cv
from PIL import ImageFont, ImageDraw, Image
from moviepy.editor import VideoFileClip
from googletrans import Translator, LANGUAGES
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from models import *
from preprocessing import *

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

language_mapping: dict[str, str] = {}


async def read_language_mapping(filename: str = "language_mapping.txt") -> None:
    with open(filename, "r") as file:
        for line in file:
            key, value = line.strip().split(":")
            language_mapping[key] = value

read_language_mapping()


async def detect_language(text: str) -> str:
    return translator.detect(text).lang


async def translate_text(text: str, src: str = 'auto', dest: str = 'en') -> str:
    return translator.translate(text=text, src=src, dest=dest).text


async def get_tesseract_language(google_translate_code: str) -> str:
    return language_mapping.get(google_translate_code, "")


async def convert_bytes_in_image(image: UploadFile) -> np.ndarray:
    content = await image.read()
    np_arr = np.frombuffer(content, np.uint8)
    return cv.imdecode(np_arr, cv.IMREAD_COLOR)


async def convert_image_in_bytes(img: np.ndarray, etx: str) -> bytes:
    _, img_encoded = cv.imencode(ext=etx, img=img)
    return img_encoded.tobytes()


async def get_image_data_from_bytes(image: UploadFile) -> Tuple[np.ndarray, str]:
    from pathlib import Path
    if isinstance(image, str):
        response = requests.get(image)
        response.raise_for_status()
        image = response.content

    img = await convert_bytes_in_image(image=image)
    etx = Path(image.filename).suffix.lower()

    return img, etx


async def get_media_data_from_bytes(media: Union[UploadFile, str]):
    from pathlib import Path

    if isinstance(media, str):
        response = requests.get(media)
        response.raise_for_status()
        media_data = response.content
        ext = Path(media).suffix.lower()
    else:
        media_data = await media.read()
        ext = Path(media.filename).suffix.lower()

    return media_data, ext


def extract_image_information(image: np.ndarray, lang: str, confidence_threshold: float = 70.0) -> Tuple[int, np.ndarray, TextData]:
    config = '--tessdata-dir tesseract/tessdata_better --oem 3 --psm 11'
    textdata = TextData(pytesseract.image_to_data(image=image, lang=lang, config=config, output_type=pytesseract.Output.DICT))
    filtered_textdata = textdata.filter_textdata(confidence_threshold)
    num_confiable_texts = len(filtered_textdata.conf)
    return num_confiable_texts, image, filtered_textdata


async def avaliable_better_image_for_reader(images: List[np.ndarray], lang: str, confidence_threshold: float = 70.0) -> Tuple[np.ndarray, dict]:
    best_image_data: dict = {'num_confiable_texts': 0, 'image': None, 'textdata': None}
    i = time()

    async def process_images_async(args: Tuple[np.ndarray, str, float]) -> Tuple[int, np.ndarray, TextData]:
        return extract_image_information(*args)

    tasks = [process_images_async((image, lang, confidence_threshold)) for image in images]
    results = await asyncio.gather(*tasks)
    filtred_textdata_list: List[TextData] = []

    for num_confiable_texts, image, textdata in results:
        if textdata:
            filtred_textdata_list.append(textdata)
            if num_confiable_texts > best_image_data['num_confiable_texts']:
                best_image_data['num_confiable_texts'] = num_confiable_texts
                best_image_data['image'] = image
                best_image_data['textdata'] = textdata

    f = time()
    print(f'tempo {round(f - i, 2)} seconds')
    best_image_data['textdata'] = TextData.merge_textdata(filtred_textdata_list, 5)
    return best_image_data['image'], best_image_data['textdata']


def apply_image_processing(image: np.ndarray) -> dict[str, np.ndarray]:

    images = {'image': image}
    images['resize'] = resize_image(images['image'], 0.5)
    images['masked'] = masked(images['image'])
    images['resize'] = resize_image(images['masked'], 2.0)

    cv2.imshow('a', images['masked'])
    if cv2.waitKey(0) and ord('q'):
        cv2.destroyAllWindows()

    images['gray'] = color_to_gray(images['resize'])
    images['invert'] = invert_colors(images['gray'])
    images['bilateral'] = bilateral_filtrage(images['invert'])
    images['equalize'] = equalize_histogram(images['bilateral'])
    # images['morph_transform'] = morphological_transform(images['equalize'])
    # images['contrast'] = enhance_contrast(images['morph_transform'])
    # images['blur'] = blur_image(images['contrast'])
    # images['threshold'] = adaptive_threshold(images['blur'])
    # images['morph_opening'] = morphological_oppening(images['threshold'])
    # images['morph_closing'] = morphological_closing(images['threshold'])
    # # images['canny'] = canny_edge_detection(images['threshold'])
    # images['thick_font'] = thick_font(images['threshold'])
    # images['thin_font'] = thin_font(images['threshold'])
    # images['noise_removal'] = noise_removal(images['threshold'])
    # images['remove_borders'] = remove_borders(images['threshold'])
    # images['desknew'] = desknew(images['threshold'])

    return images


async def process_image(image: np.ndarray, lang: str):
    images = apply_image_processing(image).values()
    better, textdata = await avaliable_better_image_for_reader(images, lang)
    return better, textdata


async def extract_text_from_image(src: str, content: np.ndarray = None) -> TextData:
    ocr_src = get_tesseract_language(src)
    _, textdata = await process_image(image=content, lang=await ocr_src)
    return TextData(textdata)


async def draw_rectangles(draw, lines):
    for line in lines:
        line_x, line_y, line_w, line_h, original_line_text = line['x'], line['y'], line['w'], line['h'], line['text']

        contains_special_character = unidecode(original_line_text) != original_line_text
        margin = 2

        if contains_special_character:
            additional_size = int(line_h / 3)
        else:
            additional_size = 0

        draw.rectangle((line_x - margin, line_y - additional_size - margin, line_w + margin * 2, line_h + line_y + additional_size + margin * 2), fill='white')


async def translate_and_draw(draw, lines, translated_lines):
    for i, line in enumerate(lines):
        x, y, h = line['x'], line['y'], line['h']

        if i < len(translated_lines):
            try:
                font = ImageFont.truetype("impact.ttf", int(h * 1.25), encoding='unic')
                translated_line = translated_lines[i]
                draw.text((x, y), translated_line, fill=0, font=font)
            except Exception as e:
                raise e


async def translate_and_replace_text_from_image(src: str, dest: str, content: np.ndarray = None):
    if src == 'auto':
        src = 'en'

    textdata = extract_text_from_image(src, content)

    img_replaced = content.copy()
    img_replaced = color_to_gray(img_replaced)

    img_pil = Image.fromarray(img_replaced)
    draw = ImageDraw.Draw(img_pil)

    paragraphs = Paragraph.get_paragraphs_from_textdata(await textdata)

    for i, paragraph in enumerate(paragraphs):
        if isinstance(paragraph, list):
            continue

        text = paragraph.text

        if text.strip() == '':
            continue

        lines = paragraph.lines
        avg_length = sum(len(line.text) for line in lines) / len(lines) if lines else 0

        translated_text = await translate_text(text, src=src, dest=dest)
        translated_lines = wrap(text=translated_text, width=avg_length)

        await draw_rectangles(draw, lines)
        await translate_and_draw(draw, lines, translated_lines)

    imgReplaced = cv.cvtColor(np.array(img_pil), cv.COLOR_GRAY2BGR)

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
    if not text:
        raise HTTPException(status_code=400, detail="Texto não fornecido.")
    if src not in LANGUAGES and src != 'auto':
        raise HTTPException(status_code=400, detail=f"Linguagem de origem não suportada: {src}")
    if dest not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Linguagem de destino não suportada: {dest}")

    try:
        translated_text = translate_text(text=text, dest=dest, src=src)
        return {"translatedText": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao traduzir texto: {str(e)}")


@app.post("/api/translate-multiple")
async def api_translate_multiple(texts: List[str] = Body(...), dest: str = Body(...), src: str = Body(...)):
    if not texts:
        raise HTTPException(status_code=400, detail="Nenhum texto fornecido.")
    if src not in LANGUAGES and src != 'auto':
        raise HTTPException(status_code=400, detail=f"Linguagem de origem não suportada: {src}")
    if dest not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Linguagem de destino não suportada: {dest}")

    try:
        translated_texts = [translate_text(text=text, dest=dest, src=src) for text in texts]
        return {"translatedTexts": translated_texts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao traduzir textos: {str(e)}")


@app.post("/api/translate-multiple-for-multiple")
async def api_translate_multiples(texts: List[str] = Body(...), destLanguages: List[str] = Body(...)):
    if not texts or not destLanguages:
        raise HTTPException(status_code=400, detail="Nenhum texto ou linguagem de destino fornecidos.")
    if len(texts) != len(destLanguages):
        raise HTTPException(status_code=400, detail="O número de textos e linguagens de destino deve ser o mesmo.")
    if any(lang not in LANGUAGES for lang in destLanguages):
        raise HTTPException(status_code=400, detail="Algumas linguagens de destino não são suportadas.")

    try:
        translations = {lang: translate_text(text=text, dest=lang) for text, lang in zip(texts, destLanguages)}
        return {"translations": translations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao traduzir textos: {str(e)}")


@app.post("/api/extract-text-from-image")
async def api_extract_text_from_image(src: str = Body(...), image: Union[UploadFile, str] = File(...)):
    try:
        if not image:
            raise HTTPException(status_code=400, detail="Nenhuma imagem fornecida.")
        if src not in LANGUAGES and src != 'auto':
            raise HTTPException(status_code=400, detail=f"Linguagem de origem não suportada: {src}")

        if isinstance(image, str):
            response = requests.get(image)
            response.raise_for_status()
            image = response.content

        img = await convert_bytes_in_image(image=image)
        textdata = await extract_text_from_image(src=src, content=img)

        paragraphs = Paragraph.get_paragraphs_from_textdata(textdata)

        return {"paragraphs": paragraphs}
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Erro ao carregar a imagem pelo link fornecido: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao extrair texto da imagem: {str(e)}")


@app.post("/api/translate-image-and-replace")
async def api_translate_and_replace_text_from_image(
    src: str = Body(...),
    dest: str = Body(...),
    image: Union[UploadFile, str] = File(...)
):
    from io import BytesIO

    try:
        if not image:
            raise HTTPException(status_code=400, detail="Nenhuma imagem fornecida.")
        if src not in LANGUAGES and src != 'auto':
            raise HTTPException(status_code=400, detail=f"Linguagem de origem não suportada: {src}")
        if dest not in LANGUAGES:
            raise HTTPException(status_code=400, detail=f"Linguagem de destino não suportada: {dest}")

        img, etx = await get_image_data_from_bytes(image=image)

        images = apply_image_processing(img).values()
        better, textdata = await avaliable_better_image_for_reader(images, await get_tesseract_language(src))

        img_replaced = await translate_and_replace_text_from_image(src=src, dest=dest, content=better)
        img_converted = await convert_image_in_bytes(etx=etx, img=img_replaced)

        return StreamingResponse(BytesIO(img_converted), media_type="image/jpeg")

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Erro ao carregar a imagem pelo link fornecido: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao carregar imagem: {str(e)}")


@app.post("/api/extract-text-from-video")
async def api_extract_text_from_video(src: str = Body(None), video: Union[UploadFile, str] = File(...)):
    import uuid
    import os

    try:
        video_data, video_ext = await get_media_data_from_bytes(video)

        video_path = f"temp_video_{uuid.uuid4()}.{video_ext}"
        with open(video_path, 'wb') as video_file:
            video_file.write(video_data)

        clip = VideoFileClip(video_path)
        frames = (frame for frame in clip.iter_frames())

        paragraphs = []
        for frame in frames:
            textdata = extract_text_from_image(src, frame)
            paragraphs.append(Paragraph.get_paragraphs_from_textdata(await textdata))

        os.remove(video_path)

        return {"paragraphs": paragraphs}
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Arquivo de vídeo não encontrado.")
    except cv.error as e:
        return JSONResponse(content={"error": f"Erro ao processar vídeo: {str(e)}"}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": f"Erro inesperado: {str(e)}"}, status_code=500)
