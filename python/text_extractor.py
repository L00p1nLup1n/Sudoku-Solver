import os

import cv2
import numpy as np
import pytesseract
from PIL import Image


def setup_tesseract():
    if not os.environ.get("TESSDATA_PREFIX"):
        os.environ["TESSDATA_PREFIX"] = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "tessdata")
        )


def preprocess_image_for_text(image_path, method="adaptive"):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if method == "adaptive":
        # Adaptive thresholding - good for varying lighting
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    elif method == "otsu":
        # Otsu's thresholding - automatic threshold selection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    elif method == "enhanced":
        # Enhanced preprocessing with multiple steps
        # 1. Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # 2. Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 3. Sharpen
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        # 4. Threshold
        blurred = cv2.GaussianBlur(sharpened, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh

    elif method == "combined":
        # Combined approach with morphological operations
        # 1. Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # 2. CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 3. Morphological gradient to enhance text edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, kernel)

        # 4. Threshold
        _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 5. Morphological closing to connect text components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        return closed

    else:
        raise ValueError(f"Unknown preprocessing method: {method}")


def upscale_image(img, scale_factor=2):
    if isinstance(img, Image.Image):
        new_width = int(img.width * scale_factor)
        new_height = int(img.height * scale_factor)
        return img.resize((new_width, new_height), Image.LANCZOS)
    else:
        # numpy array
        new_width = int(img.shape[1] * scale_factor)
        new_height = int(img.shape[0] * scale_factor)
        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


def deskew_image(img):
    # Calculate skew angle
    coords = np.column_stack(np.where(img > 0))
    if len(coords) == 0:
        return img

    angle = cv2.minAreaRect(coords)[-1]

    # Adjust angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Rotate image to deskew
    if abs(angle) > 0.5:  # Only deskew if angle is significant
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

    return img


def remove_borders(img, border_size=10):
    if len(img.shape) == 2:  # Grayscale
        h, w = img.shape
        return img[border_size : h - border_size, border_size : w - border_size]
    else:  # Color
        h, w = img.shape[:2]
        return img[border_size : h - border_size, border_size : w - border_size]


def extract_text_from_image(
    image_path,
    preprocess=True,
    method="enhanced",
    psm=3,
    lang="eng",
    upscale=True,
    deskew=True,
    remove_border=False,
):
    setup_tesseract()

    try:
        if preprocess:
            # Preprocess for better OCR
            processed_img = preprocess_image_for_text(image_path, method=method)

            # Optional: deskew
            if deskew:
                processed_img = deskew_image(processed_img)

            # Optional: remove borders
            if remove_border:
                processed_img = remove_borders(processed_img)

            pil_image = Image.fromarray(processed_img)

            # Optional: upscale for small images
            if upscale and (pil_image.width < 1000 or pil_image.height < 1000):
                pil_image = upscale_image(pil_image, scale_factor=2)
        else:
            # Use original image
            pil_image = Image.open(image_path)

            # Still apply upscaling if requested
            if upscale and (pil_image.width < 1000 or pil_image.height < 1000):
                pil_image = upscale_image(pil_image, scale_factor=2)

        # Extract text using Tesseract with optimized config
        # OEM 3: Default, based on what is available (LSTM + Legacy)
        # OEM 1: LSTM only (usually more accurate for modern documents)
        config = f"--psm {psm} --oem 1"

        if lang != "eng":
            text = pytesseract.image_to_string(pil_image, lang=lang, config=config)
        else:
            text = pytesseract.image_to_string(pil_image, config=config)

        return text.strip()

    except Exception as e:
        raise RuntimeError(f"Error during text extraction: {str(e)}")


def save_extracted_text(text, output_path):
    """Save extracted text to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Text extracted and saved to: {output_path}")


def save_preprocessed_image(image_path, output_path, method="enhanced"):
    processed = preprocess_image_for_text(image_path, method=method)
    cv2.imwrite(output_path, processed)
    print(f"Preprocessed image saved to: {output_path}")


def extract_and_save(
    input_image_path,
    output_dir="output",
    preprocess=True,
    method="enhanced",
    psm=3,
    save_preprocessed=False,
):
    # Extract text
    text = extract_text_from_image(
        input_image_path,
        preprocess=preprocess,
        method=method,
        psm=psm,
        upscale=True,
        deskew=True,
    )

    # Generate output filename
    input_filename = os.path.basename(input_image_path)
    output_filename = os.path.splitext(input_filename)[0] + ".txt"
    output_path = os.path.join(output_dir, output_filename)

    # Save text
    save_extracted_text(text, output_path)

    # Optionally save preprocessed image
    if save_preprocessed and preprocess:
        preprocessed_filename = (
            os.path.splitext(input_filename)[0] + "_preprocessed.png"
        )
        preprocessed_path = os.path.join(output_dir, preprocessed_filename)
        save_preprocessed_image(input_image_path, preprocessed_path, method=method)

    return output_path
