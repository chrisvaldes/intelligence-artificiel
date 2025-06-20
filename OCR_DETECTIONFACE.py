# -*- coding: utf-8 -*-
"""
Created on Thu May 15 05:34:05 2025
@author: YOUMBI VALDES
"""
"""
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import pytesseract
import numpy as np
import shutil
from pathlib import Path
from deepface import DeepFace
import os
import json

# Configuration de Tesseract (modifie le chemin si nécessaire)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 📁 Initialisation de l’API
app = FastAPI()

# CORS pour autoriser les appels depuis le front
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change * en domaine spécifique en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📁 Répertoires
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 📌 Initialiser le détecteur de visage OpenCV
prototxt_path = "deploy.prototxt"
caffemodel_path = "res10_300x300_ssd_iter_140000.caffemodel"
face_detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)


# 🔍 OCR
def extract_text(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return "Erreur : image non lisible"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    text = pytesseract.image_to_string(gray, lang='eng')
    return text.strip()


# 🔍 Détection flou
def detect_blur(image_path, threshold=100):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None

    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = laplacian.var()
    return variance, bool(variance < threshold)


# 🔍 Détection visage + recadrage
def detect_face(image_path, scale=1.8):
    image = cv2.imread(image_path)
    if image is None:
        return None

    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    max_area = 0
    best_face = None
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x, y, x_max, y_max = box.astype("int")
            area = (x_max - x) * (y_max - y)
            if area > max_area:
                max_area = area
                best_face = (x, y, x_max, y_max)

    if best_face is None:
        return None

    x, y, x_max, y_max = best_face
    face_w, face_h = x_max - x, y_max - y
    new_w = int(face_w * scale)
    new_h = int(face_h * scale)
    x = max(0, x - (new_w - face_w) // 2)
    y = max(0, y - (new_h - face_h) // 2)
    x_max = min(w, x + new_w)
    y_max = min(h, y + new_h)

    face_img = image[y:y_max, x:x_max]
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.equalizeHist(face_img)
    face_img = cv2.resize(face_img, (224, 224))

    output_path = image_path.replace(".jpg", "_face.jpg").replace(".jpeg", "_face.jpg")
    cv2.imwrite(output_path, face_img)
    return output_path

def format_date_for_ocr(date_str):
    """Convertit '1986-01-07' en '07.01.1986'"""
    try:
        parts = date_str.split("-")
        if len(parts) == 3:
            return f"{parts[2]}.{parts[1]}.{parts[0]}"
        return date_str
    except:
        return date_str

# ✅ Endpoint principal
@app.post("/verification-globale/")
async def verification_globale(
    donnees: str = Form(...),
    photo_client: UploadFile = File(...),
    photo_id: UploadFile = File(...),
    verso_id: UploadFile = File(...)
):
    try:
        # ✅ Décodage JSON
        user_data = json.loads(donnees)

        # 📁 Sauvegarder les fichiers
        client_path = f"{UPLOAD_FOLDER}/{photo_client.filename}"
        recto_path = f"{UPLOAD_FOLDER}/{photo_id.filename}"
        verso_path = f"{UPLOAD_FOLDER}/{verso_id.filename}"

        with open(client_path, "wb") as f:
            shutil.copyfileobj(photo_client.file, f)
        with open(recto_path, "wb") as f:
            shutil.copyfileobj(photo_id.file, f)
        with open(verso_path, "wb") as f:
            shutil.copyfileobj(verso_id.file, f)

        # 🔍 OCR sur recto
        ocr_response = extract_text(recto_path)

        # 🔍 OCR sur verso
        ocr_response_verso = extract_text(verso_path)

        # 🔀 Fusion texte recto + verso
        ocr_combined = ocr_response + "\n\n" + ocr_response_verso

        # 🔍 Détection de flou
        variance, is_blurry = detect_blur(recto_path)

        # 🔍 Visages
        face_client = detect_face(client_path)
        face_cni = detect_face(recto_path)

        if not face_client or not face_cni:
            return {"status": "error", "message": "Visage non détecté dans une image"}

        # 🔍 Comparaison de visages
        face_result = DeepFace.verify(
            face_client, face_cni,
            model_name="VGG-Face",
            distance_metric="cosine",
            enforce_detection=False
        )

        # 🔍 Correspondance OCR <-> champs
        match_score = 0 
        if user_data.get('nom', '').lower() in ocr_combined.lower():
            match_score += 1
        if user_data.get('prenom', '').lower() in ocr_combined.lower():
            match_score += 1
        if user_data.get('lieuNaissance', '').lower() in ocr_combined.lower():
            match_score += 1
 
        formatted_date = format_date_for_ocr(user_data.get('dateNaissance', ''))
        if formatted_date in ocr_combined:
            match_score += 1
        formatted_date = format_date_for_ocr(user_data.get('dateDelivrance', ''))
        if formatted_date in ocr_combined:
            match_score += 1
        formatted_date = format_date_for_ocr(user_data.get('dateExpiration+', ''))
        if formatted_date in ocr_combined:
            match_score += 1

        if face_result["verified"] and not is_blurry and match_score >= 3:
            return {
                "status": "success",
                "ocr_text": ocr_combined,
                "is_blurry": is_blurry,
                "blur_variance": variance,
                "face_verified": face_result["verified"],
                "similarity": face_result["distance"],
                "match_score": match_score,
                "payload": user_data,
                "nom": user_data.get('nom', '').lower(),
                "prenom": user_data.get('prenom', '').lower(),
                "dateNaissance": user_data.get('dateNaissance', ''),
                "dateDelivrance":user_data.get('dateDelivrance', ''),
                "dateExpiration" : user_data.get('dateExpriration', ''),
            }
        else:
            return {
                "status" : "false",
                "face_verify " : face_result["verified"],
                "is_blur " : is_blurry,
                "match_score" : match_score
                }

    except Exception as e:
        return {"status": "error", "message": str(e)}
"""