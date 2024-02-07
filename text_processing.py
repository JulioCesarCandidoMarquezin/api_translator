from models import Line, Paragraph, TextData
from PIL import ImageFont, ImageDraw, Image, ImageFilter
from preprocessing import color_to_gray, process_image
from translation import translate_text
from numpy import ndarray, array
from typing import Literal, Dict
from unidecode import unidecode
from textwrap import wrap
import cv2 as cv
import asyncio

language_mapping: Dict[str, str] = {}


def read_language_mapping(filename: str = "language_mapping.txt") -> None:
    with open(filename, "r") as file:
        for line in file:
            key, value = line.strip().split(":")
            language_mapping[key] = value


read_language_mapping()


async def get_tesseract_language(google_translate_code: str) -> str:
    return language_mapping.get(google_translate_code, "en")


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
            x1 = int(x1 + (x2 - x1 - text_w) / 2)
        elif align == 'right':
            x1 = x2 - text_w
        else:
            x1 = int(x1 + (x2 - x1 - text_w) / 2)

        y_translation = int(y1 + (y2 - y1 - line_h * len(text.split('\n'))) / 2)

        text_mask = Image.new('L', (x2 - x1, y2 - y1), 0)
        text_draw = ImageDraw.Draw(text_mask)
        text_draw.text((0, 0), text, fill=255, font=font)

        text_mask = text_mask.filter(ImageFilter.GaussianBlur(radius=5))

        draw.bitmap((x1, y_translation), text_mask, fill=0)

    except Exception as e:
        raise Exception(f"Error drawing text: {e}")


async def extract_text_from_image(src: str, image: ndarray) -> TextData:
    ocr_src = get_tesseract_language(src)
    _, textdata = await process_image(image=image, lang=await ocr_src)
    return textdata


async def translate_and_replace_text_from_image(src: str, dest: str, image: ndarray = None, align: Literal['left', 'center', 'right'] = 'center') -> ndarray:
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

    imgReplaced = cv.cvtColor(array(img_pil), cv.COLOR_GRAY2BGR)

    return imgReplaced
