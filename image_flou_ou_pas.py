"""
import cv2
import pytesseract
import numpy as np

# Chemin de Tesseract (à modifier selon ton installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Pour Windows

def detect_blur(image_path, threshold=300):
    # Détecte si une image est floue en utilisant la variance du Laplacien.
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Erreur : Impossible de charger l'image.")
        return None, None

    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = laplacian.var()
    return variance, variance < threshold

def extract_text(image_path):
    # Utilise Tesseract OCR pour extraire le texte de l'image.
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Améliorer la netteté avec un filtre
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    text = pytesseract.image_to_string(gray, lang='eng')

    return text

# Tester avec l'image
image_path = "image/v_pass.jpeg"  # Remplace avec le bon chemin
variance, is_blurry = detect_blur(image_path)
extracted_text = extract_text(image_path)

# Afficher les résultats
print(f"Variance du Laplacien : {variance:.2f}")
print("⚠️ L'image est floue !" if is_blurry else "✅ L'image est nette !")
print("\n🔍 Texte extrait :\n", extracted_text)
"""

from fastapi import FastAPI, UploadFile, File
import cv2
import pytesseract
import numpy as np
import shutil
from pathlib import Path

# Initialisation de l'API
app = FastAPI()

# Configuration de Tesseract (à adapter selon ton OS)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_blur(image_path, threshold=9000):
    """Détecte si une image est floue en utilisant la variance du Laplacien."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None  # Erreur de lecture

    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = laplacian.var()
    return variance, bool(variance < threshold)  # ✅ Conversion en bool natif

def extract_text(image_path):
    """Utilise Tesseract OCR pour extraire le texte de l'image."""
    image = cv2.imread(str(image_path))
    if image is None:
        return "Erreur : Image non lisible."

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    text = pytesseract.image_to_string(gray, lang='eng')

    return text.strip()

@app.post("/detect_blur/")
async def detect_blur_api(image: UploadFile = File(...)):
    """Endpoint pour détecter le flou et extraire le texte d'une image."""
    temp_path = Path(f"temp_{image.filename}")
    with temp_path.open("wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    variance, is_blurry = detect_blur(temp_path)
    extracted_text = extract_text(temp_path)

    temp_path.unlink()  # Supprime l'image temporaire

    return {
        "variance": variance,
        "is_blurry": is_blurry,  # ✅ Booléen natif Python
        "extracted_text": extracted_text
    }
