from text_processing import extract_text_from_image, translate_and_replace_text_from_image
from preprocessing import image_data_from_bytes, image_in_bytes, bytes_in_image, media_data_from_bytes, download_image
from fastapi import Body, File, UploadFile, HTTPException, WebSocket, FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from requests.exceptions import RequestException
from moviepy.editor import VideoFileClip
from typing import List, Union, Literal
from models import Paragraph
from translation import *
from io import BytesIO
import asyncio
import base64
import json
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/supported-languages")
async def api_supported_languages():
    try:
        return {"supported_languages": LANGUAGES}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving supported languages: {str(e)}")


@app.post("/api/detect-language")
async def api_detect_language(text: str = Body(...)):
    try:
        if not text:
            raise HTTPException(status_code=400, detail="Text not provided.")

        detected_language = await detect_language(text)
        return {"detected_language": detected_language}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting language: {str(e)}")


@app.post("/api/translate-text")
async def api_translate_text(
    text: str = Body(...),
    src: str = Body('auto'),
    dest: str = Body('en')
):
    if not text:
        raise HTTPException(status_code=400, detail="Text not provided.")
    if src not in LANGUAGES and src != 'auto':
        raise HTTPException(status_code=400, detail=f"Unsupported source language: {src}")
    if dest not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported destination language: {dest}")

    try:
        translated_text = await translate_text(text=text, dest=dest, src=src)
        return JSONResponse(content={'translated_text': translated_text}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error translating text: {str(e)}")


@app.post("/api/translate-multiple")
async def api_translate_multiple(
    texts: List[str] = Body(...), dest: str = Body('en'), src: str = Body('auto')
):
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided.")
    if src not in LANGUAGES and src != 'auto':
        raise HTTPException(status_code=400, detail=f"Unsupported source language: {src}")
    if dest not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported destination language: {dest}")

    try:
        translated_texts = await asyncio.gather(*[translate_text(text=text, dest=dest, src=src) for text in texts])
        return JSONResponse(content={'translated_texts': translated_texts}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error translating texts: {str(e)}")


@app.post("/api/translate-multiple-for-multiple")
async def api_translate_multiple_for_multiple(
    texts: List[str] = Body(...), dest_languages: List[str] = Body(...)
):
    if not texts or not dest_languages:
        raise HTTPException(status_code=400, detail="No texts or destination languages provided.")
    if len(texts) != len(dest_languages):
        raise HTTPException(
            status_code=400, detail="The number of texts and destination languages must be the same."
        )
    if any(lang not in LANGUAGES for lang in dest_languages):
        raise HTTPException(status_code=400, detail="Some destination languages are not supported.")

    try:

        translations = await asyncio.gather(*(translate_text(text, dest=dest) for text, dest in zip(texts, dest_languages)))
        return JSONResponse(content={'translations': translations}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error translating texts: {str(e)}")


@app.post("/api/extract-text-from-image")
async def api_extract_text_from_image(
    src: str = Body(...),
    image: Union[UploadFile, str] = File(...),
):
    try:
        if not image:
            raise HTTPException(status_code=400, detail="No image provided.")
        if src not in LANGUAGES and src != 'auto':
            raise HTTPException(status_code=400, detail=f"Unsupported source language: {src}")

        image = await bytes_in_image(image_bytes=await image.read())

        textdata = await extract_text_from_image(src=src, image=image)
        paragraphs = Paragraph.get_paragraphs_from_textdata(textdata)

        paragraphs = [paragraph.to_dict() for paragraph in paragraphs]

        return JSONResponse(content={'paragraphs': paragraphs}, status_code=200)

    except RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error loading image from the provided link: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text from image: {str(e)}")


@app.websocket("/api/extract-texts-from-images")
async def api_extract_texts_from_images(websocket: WebSocket):
    await websocket.accept()

    try:
        data = await websocket.receive_text()
        data = json.loads(data)

        src = data.get('src')
        images = data.get('images')

        if not images:
            await websocket.send_text("No images provided.")
            return
        if src not in LANGUAGES and src != 'auto':
            await websocket.send_text(f"Unsupported source language: {src}")
            return

        total_images = len(images)

        for index, image in enumerate(images):
            if image.strip().lower().startswith('http'):
                image = await download_image(image)
            else:
                image = base64.b64decode(image)

            image = await bytes_in_image(image)
            textdata = await extract_text_from_image(src=src, image=image)
            paragraphs = Paragraph.get_paragraphs_from_textdata(textdata)
            paragraphs = [paragraph.to_dict() for paragraph in paragraphs]

            percentage_complete = (index + 1) / total_images * 100
            await websocket.send_json({
                "status": "partial",
                "result": {"paragraphs": paragraphs},
                "percentage_complete": percentage_complete
            })

        await websocket.send_json({"status": "complete", "percentage_complete": 100})

    except RequestException as e:
        await websocket.send_text(f"Error loading image from the provided link: {str(e)}")
    except Exception as e:
        await websocket.send_text(f"Error extracting text from images: {str(e)}")


@app.post("/api/translate-image-and-replace")
async def api_translate_and_replace_text_from_image(
        src: str = Body(...),
        dest: str = Body(...),
        align: Literal['left', 'center', 'right'] = Body('center'),
        image: Union[UploadFile, str] = File(...)
):
    try:
        if not image:
            raise HTTPException(status_code=400, detail="No image provided.")
        if src not in LANGUAGES and src != 'auto':
            raise HTTPException(status_code=400, detail=f"Unsupported source language: {src}")
        if dest not in LANGUAGES:
            raise HTTPException(status_code=400, detail=f"Unsupported destination language: {dest}")

        image_data, etx = await image_data_from_bytes(image_bytes=await image.read())

        replaced_image = await translate_and_replace_text_from_image(src=src, dest=dest, image=image_data, align=align)
        converted_image = await image_in_bytes(etx=etx, img=replaced_image)

        return StreamingResponse(BytesIO(converted_image), media_type="image/jpeg")

    except RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error loading the image from the provided link: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading the image: {str(e)}")


@app.websocket("/api/extract-text-from-media")
async def api_extract_text_from_media(websocket: WebSocket):
    await websocket.accept()

    try:
        data = await websocket.receive_text()
        data = json.loads(data)

        src = data.get('src')

        if src not in LANGUAGES:
            await websocket.send_text(f"Unsupported source language: {src}")
            return

        frame = 0
        while True:
            chunk = await websocket.receive_bytes()
            if chunk == b"EOF":
                break

            frame += 1

            image = await bytes_in_image(image_bytes=chunk)

            textdata = await extract_text_from_image(src=src, image=image)
            paragraphs = Paragraph.get_paragraphs_from_textdata(textdata)

            paragraphs = [paragraph.to_dict() for paragraph in paragraphs]

            await websocket.send_json({
                "status": "partial",
                "frame": frame,
                "result": {"paragraphs": paragraphs},
            })

        await websocket.send_json({"status": "complete"})

    except FileNotFoundError:
        await websocket.send_text("Media file not found.")
    except Exception as e:
        await websocket.send_text(f"Unexpected error: {str(e)}")


@app.websocket("/api/translate-and-replace-from-media")
async def api_translate_and_replace_from_media(websocket: WebSocket):
    await websocket.accept()

    try:
        data = await websocket.receive_text()
        data = json.loads(data)

        src = data.get('src')
        dest = data.get('dest')
        align = data.get('align')
        media = data.get('media')

        if not media:
            await websocket.send_text("No media provided.")
            return
        if src not in LANGUAGES:
            await websocket.send_text(f"Unsupported source language: {src}")
            return
        if dest not in LANGUAGES:
            await websocket.send_text(f"Unsupported destination language: {dest}")
            return

        media_data, media_ext = await media_data_from_bytes(media)

        media_path = f"temp_media_{uuid.uuid4()}.{media_ext}"
        with open(media_path, 'wb') as media_file:
            media_file.write(media_data)

        clip = VideoFileClip(media_path)
        total_frames = int(clip.fps * clip.duration)

        async def process_frame(frame, current_frame):
            replaced_frame = await translate_and_replace_text_from_image(src=src, dest=dest, image=frame, align=align)
            await websocket.send_json({
                "status": "partial",
                "result": {"frame": replaced_frame},
                "percentage_complete": (current_frame + 1) / total_frames * 100
            })

        await asyncio.gather(*(process_frame(frame, current_frame) for current_frame, frame in enumerate(clip.iter_frames())))

        os.remove(media_path)

        await websocket.send_json({"status": "complete", "percentage_complete": 100})

    except FileNotFoundError:
        await websocket.send_text("Media file not found.")
    except Exception as e:
        await websocket.send_text(f"Unexpected error: {str(e)}")


if __name__ == '__main__':
    import os
    os.system('uvicorn main:app')
