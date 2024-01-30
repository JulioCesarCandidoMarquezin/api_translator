import cv2
import numpy as np
from pytesseract import pytesseract

pytesseract.tesseract_cmd = 'tesseract/tesseract.exe'


def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w * scale), int(h * scale)))


def color_to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) > 2:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def blur_image(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (5, 5), 0)


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(image)


def bilateral_filtrage(image: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(image, 9, 75, 75)


def canny_edge_detection(image: np.ndarray) -> np.ndarray:
    return cv2.Canny(image, 50, 150)


def invert_colors(image: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(image)


def distance_transform(image: np.ndarray) -> np.ndarray:
    dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    return sure_fg


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def morphological_transform(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    eroded_img = cv2.erode(image, kernel, iterations=1)
    dilated_img = cv2.dilate(image, kernel, iterations=1)
    return dilated_img


def morphological_oppening(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)


def morphological_closing(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)


def adjust_contrast(image: np.ndarray) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=1.5, beta=50)


def noise_removal(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)


def thin_font(image: np.ndarray) -> np.ndarray:
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)


def thick_font(image: np.ndarray) -> np.ndarray:
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)


def getSkewAngle(image: np.ndarray) -> float:
    new_image = image.copy()
    gray = color_to_gray(new_image)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)

    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)
    angle = minAreaRect[-1]

    if angle < -45:
        angle = 90 * angle
    return -1 * angle


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    new_image = image.copy()
    (h, w) = new_image.shape[:2]
    center = (h // 2, w // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_image = cv2.warpAffine(new_image, m, (w, h), cv2.INTER_CUBIC, cv2.BORDER_REPLICATE)
    return new_image


def desknew(image: np.ndarray) -> np.ndarray:
    angle = getSkewAngle(image)
    return rotate_image(image, angle)


def remove_borders(image: np.ndarray) -> np.ndarray:
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.contourArea(c))
    contour = contours[-1]
    x, y, w, h = cv2.boundingRect(contour)
    crop = image[y: y + h, x: x + w]
    return crop


def get_textareas_mask(image: np.ndarray, min_contour_percent: float = 1.0, max_contour_percent: float = 50.0) -> np.ndarray:
    gray = color_to_gray(image)
    invert = invert_colors(gray)
    blur = cv2.GaussianBlur(invert, (7, 7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    open = canny_edge_detection(thresh)

    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    hist = equalize_histogram(thresh)

    contours, _ = cv2.findContours(hist, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = image.shape[0] * image.shape[1]

    resolution_factor = image_area / (1920 * 1080)

    min_contour_area = int(min_contour_percent / 100 * image_area * resolution_factor)
    max_contour_area = int(max_contour_percent / 100 * image_area * resolution_factor)

    filtered_contours = [cnt for cnt in contours if min_contour_area < cv2.contourArea(cnt) < max_contour_area]

    for contour in filtered_contours:
        area_contour = cv2.contourArea(contour)
        percent_area = (area_contour / image_area) * 100

        x, y, w, h = cv2.boundingRect(contour)
        roi = hist[y:y + h, x:x + w]
        cv2.imshow('roi', roi)
        if cv2.waitKey(0) and ord('q'):
            cv2.destroyAllWindows()

    mask = np.zeros_like(image)

    cv2.drawContours(mask, filtered_contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    result_image = np.where(mask == 0, 0, 255)
    return result_image.astype(np.uint8)
