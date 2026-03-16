import pytesseract
from PIL import Image
import numpy as np
import cv2

def get_text_from_image(image: Image) -> str:
    img1 = np.array(image)
    g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    b = cv2.threshold(g, 130, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    i = cv2.bitwise_not(b)
    text = pytesseract.image_to_string(i, lang='eng')
    return text