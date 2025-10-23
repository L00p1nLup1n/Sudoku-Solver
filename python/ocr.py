"""OCR wrapper using pytesseract. Exposes a function that accepts PIL images or numpy arrays (28x28) and returns a digit or 0."""        
def ocr_image_to_digit(image):
    # image: PIL.Image or numpy array
    import os
    if not os.environ.get("TESSDATA_PREFIX"):
        os.environ["TESSDATA_PREFIX"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tessdata"))

    try:
        from PIL import Image
        import pytesseract
    except Exception as e:
        raise RuntimeError("OCR dependencies not installed: " + str(e))

    if not hasattr(image, 'mode'):
        # assume numpy array
        import numpy as np
        from PIL import Image
        image = Image.fromarray(image)

    config = "--psm 10 --oem 3 -c tessedit_char_whitelist=123456789"
    raw = pytesseract.image_to_string(image, config=config)
    import re
    digits = re.sub(r'[^1-9]', '', raw)
    return int(digits) if digits else 0
