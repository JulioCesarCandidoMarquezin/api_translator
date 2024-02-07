from typing import Tuple, Union, List, Dict
from pytesseract import pytesseract
from fastapi import UploadFile
from models import TextData
from io import BytesIO
import numpy as np
import cv2 as cv
import requests
import asyncio
import imghdr
import base64
import magic

pytesseract.tesseract_cmd = 'tesseract/tesseract.exe'
cv.setUseOptimized(True)
cv.setNumThreads(8)
cv.ocl.setUseOpenCL(True)


async def download_image(url: str) -> bytes:
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to download image from {url}, status code: {response.status_code}")


async def download_media(url: str):
    pass


async def bytes_in_image(image_bytes: bytes) -> np.ndarray:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    return cv.imdecode(np_arr, cv.IMREAD_COLOR)


async def image_in_bytes(img: np.ndarray, etx: str) -> bytes:
    if etx:
        etx = '.' + etx
    else:
        etx = '.jpeg'

    _, img_encoded = cv.imencode(ext=etx, img=img)
    return img_encoded.tobytes()


async def image_data_from_bytes(image_bytes: bytes) -> Tuple[np.ndarray, str]:
    img_array = cv.imdecode(np.frombuffer(image_bytes, np.uint8), cv.IMREAD_COLOR)
    image_extension = imghdr.what(BytesIO(image_bytes))

    return img_array, image_extension


async def media_data_from_bytes(media: bytes) -> Tuple[bytes, str]:
    media_base64 = base64.b64decode(media)

    mime = magic.Magic(mime=True)
    media_ext = mime.from_buffer(media_base64)

    return media_base64, media_ext


def apply_image_processing(image: np.ndarray) -> Dict[str, np.ndarray]:

    images = {}
    # images['masked'] = masked(image)
    # images = {'image': image}
    # images['resize'] = resize_image(images['image'], 0.5)
    # images['resize'] = resize_image(images['masked'], 2.0)

    images['gray'] = color_to_gray(image)
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


def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    h, w = image.shape[:2]
    return cv.resize(image, (int(w * scale), int(h * scale)))


def color_to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) > 2:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image


def blur_image(image: np.ndarray) -> np.ndarray:
    return cv.GaussianBlur(image, (5, 5), 0)


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    return cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2)


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    return cv.equalizeHist(image)


def bilateral_filtrage(image: np.ndarray) -> np.ndarray:
    return cv.bilateralFilter(image, 9, 75, 75)


def canny_edge_detection(image: np.ndarray) -> np.ndarray:
    return cv.Canny(image, 50, 150)


def invert_colors(image: np.ndarray) -> np.ndarray:
    return cv.bitwise_not(image)


def distance_transform(image: np.ndarray) -> np.ndarray:
    dist_transform = cv.distanceTransform(image, cv.DIST_L2, 5)
    _, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    return sure_fg


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def morphological_transform(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    eroded_img = cv.erode(image, kernel, iterations=1)
    dilated_img = cv.dilate(image, kernel, iterations=1)
    return dilated_img


def morphological_oppening(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv.morphologyEx(image, cv.MORPH_OPEN, kernel, iterations=2)


def morphological_closing(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv.morphologyEx(image, cv.MORPH_CLOSE, kernel, iterations=2)


def adjust_contrast(image: np.ndarray) -> np.ndarray:
    return cv.convertScaleAbs(image, alpha=1.5, beta=50)


def noise_removal(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image = cv.dilate(image, kernel, iterations=1)
    image = cv.erode(image, kernel, iterations=1)
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    image = cv.medianBlur(image, 3)
    return image


def thin_font(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image = cv.bitwise_not(image)
    image = cv.erode(image, kernel, iterations=1)
    image = cv.bitwise_not(image)
    return image


def thick_font(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image = cv.bitwise_not(image)
    image = cv.dilate(image, kernel, iterations=1)
    image = cv.bitwise_not(image)
    return image


def getSkewAngle(image: np.ndarray) -> float:
    new_image = image.copy()
    gray = color_to_gray(new_image)
    blur = cv.GaussianBlur(gray, (9, 9), 0)
    thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 5))
    dilate = cv.dilate(thresh, kernel, iterations=2)

    contours, hierarchy = cv.findContours(dilate, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda contour: cv.contourArea(contour), reverse=True)

    largestContour = contours[0]
    minAreaRect = cv.minAreaRect(largestContour)
    angle = minAreaRect[-1]

    if angle < -45:
        angle = 90 * angle
    return -1 * angle


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    new_image = image.copy()
    (h, w) = new_image.shape[:2]
    center = (h // 2, w // 2)
    m = cv.getRotationMatrix2D(center, angle, 1.0)
    new_image = cv.warpAffine(new_image, m, (w, h), cv.INTER_CUBIC, cv.BORDER_REPLICATE)
    return new_image


def desknew(image: np.ndarray) -> np.ndarray:
    angle = getSkewAngle(image)
    return rotate_image(image, angle)


def remove_borders(image: np.ndarray) -> np.ndarray:
    contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv.contourArea(c))
    contour = contours[-1]
    x, y, w, h = cv.boundingRect(contour)
    crop = image[y: y + h, x: x + w]
    return crop


def masked(image: np.ndarray, min_contour_percent: float = 10.0, max_contour_percent: float = 95.0) -> np.ndarray:
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    gray = color_to_gray(image)
    blur = cv.GaussianBlur(gray, (7, 7), 0)
    thresh = adaptive_threshold(blur)
    erode = thin_font(thresh, kernel)
    dilate = thick_font(erode, kernel)
    open = morphological_transform(dilate, kernel)
    hist = equalize_histogram(open)
    invert = invert_colors(hist)

    contours, _ = cv.findContours(invert, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    image_area = image.shape[0] * image.shape[1]

    resolution_factor = image_area / (1920 * 1080)

    min_contour_area = int(min_contour_percent / 100 * image_area * resolution_factor)
    max_contour_area = int(max_contour_percent / 100 * image_area * resolution_factor)

    filtered_contours = [cnt for cnt in contours if min_contour_area < cv.contourArea(cnt) < max_contour_area]

    mask = np.zeros_like(gray, dtype=np.uint8)

    cv.drawContours(mask, filtered_contours, -1, (255, 255, 255), thickness=cv.FILLED)

    result_image = cv.bitwise_and(image, image, mask=mask)

    return result_image
