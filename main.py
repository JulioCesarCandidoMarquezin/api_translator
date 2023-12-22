from fastapi import FastAPI, UploadFile, File, HTTPException
from moviepy.editor import VideoFileClip
import cv2 as cv
from pytesseract import pytesseract as pt
from googletrans import Translator
import numpy as np
from typing import List
import os

app = FastAPI()

@app.get("/api/supported-languages")
def supported_languages():
    try:
        translator = Translator()
        languages = translator.detect('dummy_text').lang
        return {"supported_languages": languages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao obter idiomas suportados: {str(e)}")

@app.post("/api/detect-language")
async def detect_language(text: str):
    try:
        translator = Translator()
        detected_language = translator.detect(text).lang
        return {"detected_language": detected_language}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao detectar idioma: {str(e)}")

@app.post("/api/translate-text")
async def translate_text(text: str, source_language: str = 'auto', target_language: str = 'eng'):
    try:
        translator = Translator()
        translated_text = translator.translate(text=text, src=source_language, dest=target_language).text
        return {"translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao traduzir texto: {str(e)}")

@app.post("/api/translate-multiple")
async def translate_multiple(texts: List[str], target_language: str):
    try:
        translator = Translator()
        translated_texts = [translator.translate(text, dest=target_language).text for text in texts]
        return {"translated_texts": translated_texts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao traduzir textos: {str(e)}")

@app.post("/api/translate-multiple-for-multiple")
async def translate_multiple(texts: List[str], target_languages: List[str]):
    try:
        translator = Translator()
        translations = {lang: translator.translate(text, dest=lang).text for text, lang in zip(texts, target_languages)}
        return {"translations": translations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao traduzir textos: {str(e)}")

@app.post("/api/extract-text-from-image")
async def extract_text_from_image(lang: str = None, image: UploadFile = File(...)):
    try:
        content = await image.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_COLOR)

        gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(gray_image, 25, 150, cv.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        dilatation = cv.erode(binary, kernel, iterations=1)

        extracted_text = pt.image_to_string(dilatation, lang=lang)
        text_positions = pt.image_to_boxes(dilatation, lang=lang)

        return {"extracted_text": extracted_text.strip(), "text_positions": text_positions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao extrair texto da imagem: {str(e)}")

@app.post("/api/extract-texts-from-images")
async def extract_texts_from_images(lang: str = None, images: List[UploadFile] = File(...)):
    try:
        extracted_texts = []
        extracted_texts_positions = []
        for image in images:
            content = await image.read()
            nparr = np.frombuffer(content, np.uint8)

            img = cv.imdecode(nparr, cv.IMREAD_COLOR)
            gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            _, binary = cv.threshold(gray_image, 25, 150, cv.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            dilatation = cv.erode(binary, kernel, iterations=1)

            extracted_text = pt.image_to_string(dilatation, lang=lang)
            extracted_texts_position = pt.image_to_boxes(dilatation, lang=lang)

            extracted_texts.append(extracted_text.strip())
            extracted_texts_positions.append(extracted_texts_position)
        return {"extracted_texts": extracted_texts, "extracted_texts_positions": extracted_texts_positions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao extrair texto das imagens: {str(e)}")

@app.post("/api/extract-text-from-video")
async def extract_text_from_video(lang: str = None, video: UploadFile = File(...)):
    try:
        # Salvar o vídeo temporariamente
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as video_file:
            video_file.write(video.file.read())

        # Extrair legendas do vídeo usando a biblioteca moviepy
        clip = VideoFileClip(video_path)
        frames = [frame for frame in clip.iter_frames()]

        extracted_texts = []
        for frame in frames:
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            _, binary = cv.threshold(gray_frame, 25, 150, cv.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            dilatation = cv.erode(binary, kernel, iterations=1)

            extracted_text = pt.image_to_string(dilatation, lang=lang)
            extracted_texts.append(extracted_text.strip())

        # Remover o vídeo temporário após a extração
        os.remove(video_path)

        return {"extracted_texts": extracted_texts}
    except Exception:
        return {"error": f"Erro ao extrair texto do vídeo"}