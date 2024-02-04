from textwrap import wrap
from typing import Literal
import asyncio

import pytesseract
from unidecode import unidecode

import imageio
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


def read_language_mapping(filename: str = "language_mapping.txt") -> None:
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


async def get_media_data_from_bytes(media: Union[UploadFile, str]) -> Tuple[bytes, str]:
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
    textdata = textdata.filter_textdata(confidence_threshold)
    num_confiable_texts = len(textdata.conf)
    return num_confiable_texts, image, textdata


async def avaliable_better_image_for_reader(images: List[np.ndarray], lang: str, confidence_threshold: float = 70.0) -> Tuple[np.ndarray, TextData]:
    best_image_data: dict = {'num_confiable_texts': 0, 'image': None, 'textdata': None}

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

    best_image_data['textdata'] = TextData.merge_textdata(filtred_textdata_list, 5)
    return best_image_data['image'], best_image_data['textdata']


def apply_image_processing(image: np.ndarray) -> Dict[str, np.ndarray]:

    images = {}
    images['masked'] = masked(image)
    # images = {'image': image}
    # images['resize'] = resize_image(images['image'], 0.5)
    # images['resize'] = resize_image(images['masked'], 2.0)

    images['gray'] = color_to_gray(images['masked'])
    # images['invert'] = invert_colors(images['gray'])
    # images['bilateral'] = bilateral_filtrage(images['invert'])
    # images['equalize'] = equalize_histogram(images['bilateral'])
    # images['morph_transform'] = morphological_transform(images['equalize'])
    # images['contrast'] = enhance_contrast(images['morph_transform'])
    # images['blur'] = blur_image(images['contrast'])
    # images['threshold'] = adaptive_threshold(images['blur'])
    # images['morph_opening'] = morphological_oppening(images['threshold'])
    # images['morph_closing'] = morphological_closing(images['threshold'])
    # images['canny'] = canny_edge_detection(images['threshold'])
    # images['thick_font'] = thick_font(images['threshold'])
    # images['thin_font'] = thin_font(images['threshold'])
    # images['noise_removal'] = noise_removal(images['threshold'])
    # images['remove_borders'] = remove_borders(images['threshold'])
    # images['desknew'] = desknew(images['threshold'])

    return images


async def process_image(image: np.ndarray, lang: str) -> Tuple[np.ndarray, TextData]:
    images = apply_image_processing(image).values()
    better, textdata = await avaliable_better_image_for_reader(images, lang)
    return better, textdata


async def extract_text_from_image(src: str, image: np.ndarray = None) -> TextData:
    ocr_src = get_tesseract_language(src)
    _, textdata = await process_image(image=image, lang=await ocr_src)
    return textdata


def draw_rectangle(draw: ImageDraw.ImageDraw, line: Line) -> None:
    x1, y1, x2, y2 = line.get_coordinates()
    line_text = line.text

    contains_special_character = unidecode(line_text) != line_text
    margin = 2

    if contains_special_character:
        additional_size = int((y2 - y1) / 3)
    else:
        additional_size = 0

    x1 = x1 - margin
    y1 = y1 - additional_size - margin
    x2 = x2 + margin * 2
    y2 = y2 + additional_size + margin * 2

    draw.rectangle((x1, y1, x2, y2), fill='white')


def draw_text(draw: ImageDraw.ImageDraw, line: Line, align: Literal['left', 'center', 'right'] = 'center') -> None:
    x1, y1, x2, y2 = line.get_coordinates()

    try:
        text = line.text

        line_h = int(y2 - y1)
        font_size = int(line_h * 1.25)
        font = ImageFont.truetype("arial.ttf", font_size, encoding='unic')

        text_bbox = draw.textbbox((x1, y1), text, font=font)

        text_w = int(text_bbox[2] - text_bbox[0])

        if align == 'left':
            x1 = x1
        elif align == 'center':
            x1 = x1 + (x2 - x1 - text_w) / 2
        elif align == 'right':
            x1 = x2 - text_w
        else:
            x1 = x1 + (x2 - x1 - text_w) / 2

        y_translation = y1 + (y2 - y1 - line_h * len(text.split('\n'))) / 2

        draw.text((x1, y_translation), text, fill=0, font=font, align=align)

    except Exception as e:
        raise Exception(f"Erro ao desenhar texto: {e}")


async def translate_and_replace_text_from_image(src: str, dest: str, image: np.ndarray = None, align: Literal['left', 'center', 'right'] = 'center') -> np.ndarray:
    if src == 'auto':
        src = 'en'

    textdata = extract_text_from_image(src, image)

    img_replaced = image.copy()
    img_replaced = color_to_gray(img_replaced)

    img_pil = Image.fromarray(img_replaced)
    draw = ImageDraw.Draw(img_pil)

    paragraphs = Paragraph.get_paragraphs_from_textdata(await textdata)

    translation_tasks = []

    for i, paragraph in enumerate(paragraphs):
        if isinstance(paragraph, list):
            continue

        translation_tasks.append(translate_text(paragraph.text, src, dest))

    translated_paragraphs = await asyncio.gather(*translation_tasks)

    for i, paragraph in enumerate(paragraphs):
        if isinstance(paragraph, list):
            continue

        paragraph.text = translated_paragraphs[i]
        text = paragraph.text

        lines = paragraph.lines
        avg_length = int(len(text) / len(paragraph.lines))

        translated_lines = wrap(text=text, width=avg_length)

        lines_h = int((paragraph.y2 - paragraph.y1) / len(translated_lines))
        lines_h_margin = int(lines_h * 0.15)

        translated_lines = [
            Line(paragraph.x1, paragraph.y1 + i * lines_h, paragraph.x2, paragraph.y1 + (i + 1) * lines_h - lines_h_margin,
                 translated_line) for i, translated_line in enumerate(translated_lines)]

        for line in lines:
            draw_rectangle(draw, line)

        for translated_line in translated_lines:
            draw_text(draw, translated_line, align)

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

        image = await convert_bytes_in_image(image=image)
        textdata = await extract_text_from_image(src=src, image=image)

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
    align: str = Body(...),
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

        image, etx = await get_image_data_from_bytes(image=image)

        img_replaced = await translate_and_replace_text_from_image(src=src, dest=dest, image=image, align=align)
        img_converted = await convert_image_in_bytes(etx=etx, img=img_replaced)

        return StreamingResponse(BytesIO(img_converted), media_type="image/jpeg")

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Erro ao carregar a imagem pelo link fornecido: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao carregar imagem: {str(e)}")


@app.post("/api/extract-text-from-media")
async def api_extract_text_from_media(src: str = Body(None), media: Union[UploadFile, str] = File(...)):
    import uuid
    import os
    
    try:
        if not media:
            raise HTTPException(status_code=400, detail="Nenhuma imagem fornecida.")
        if src not in LANGUAGES:
            raise HTTPException(status_code=400, detail=f"Linguagem de origem não suportada: {src}")
        
        media_data, media_ext = await get_media_data_from_bytes(media)

        media_path = f"temp_media_{uuid.uuid4()}.{media_ext}"
        with open(media_path, 'wb') as media_file:
            media_file.write(media_data)

        clip = VideoFileClip(media_path)
        frames = (frame for frame in clip.iter_frames())

        paragraphs = []
        for frame in frames:
            textdata = extract_text_from_image(src, frame)
            paragraphs.append(Paragraph.get_paragraphs_from_textdata(await textdata))

        os.remove(media_path)

        return {"paragraphs": paragraphs}
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Arquivo de vídeo não encontrado.")
    except cv.error as e:
        return JSONResponse(content={"error": f"Erro ao processar vídeo: {str(e)}"}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": f"Erro inesperado: {str(e)}"}, status_code=500)


@app.post("/api/translate-and-replace-from-media")
async def translate_and_replace_from_media(
    src: str = Body(None),
    dest: str = Body(None),
    align: str = Body(...),
    media: Union[UploadFile, str] = File(...)
):
    import uuid
    import os

    try:
        if not media:
            raise HTTPException(status_code=400, detail="Nenhuma mídia fornecida.")
        if src not in LANGUAGES:
            raise HTTPException(status_code=400, detail=f"Linguagem de origem não suportada: {src}")
        if dest not in LANGUAGES:
            raise HTTPException(status_code=400, detail=f"Linguagem de destino não suportada: {dest}")

        media_data, media_ext = await get_media_data_from_bytes(media)

        media_path = f"temp_media_{uuid.uuid4()}.{media_ext}"
        with open(media_path, 'wb') as media_file:
            media_file.write(media_data)

        clip = VideoFileClip(media_path)

        async def process_frame(frame):
            replaced_frame = await translate_and_replace_text_from_image(src=src, dest=dest, image=frame, align=align)
            return replaced_frame

        replaced_frames: List[np.ndarray] = await asyncio.gather(*(process_frame(frame) for frame in clip.iter_frames()))

        output_video_path = f"output_video_{uuid.uuid4()}.mp4"
        imageio.mimsave(output_video_path, replaced_frames, fps=clip.fps)

        os.remove(media_path)

        return StreamingResponse(open(output_video_path, 'rb'), media_type="video/mp4")
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Arquivo de mídia não encontrado.")
    except cv.error as e:
        return JSONResponse(content={"error": f"Erro ao processar mídia: {str(e)}"}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": f"Erro inesperado: {str(e)}"}, status_code=500)


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, port=80)
