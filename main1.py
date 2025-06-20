 

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import pytesseract
import numpy as np
import shutil
import os
import json
import uuid
from deepface import DeepFace
from predict_piece_type import predict_piece_types
from rapidfuzz import fuzz  # ✅ Import ajouté


pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

app = FastAPI()

origins = ["http://localhost:4200"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

prototxt_path = "deploy.prototxt"
caffemodel_path = "res10_300x300_ssd_iter_140000.caffemodel"
face_detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

def allowed_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png"))

def extract_text(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return "Erreur : image non lisible"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return pytesseract.image_to_string(gray, lang='eng').strip().lower()

def detect_blur(image_path, threshold=41):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = laplacian.var()
    return variance, variance < threshold

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


def format_date_for_ocr(date_input):
    try:
        date_str = str(date_input)
        sep = "/" if "/" in date_str else "-" if "-" in date_str else None
        if sep:
            parts = date_str.split(sep)
            print("parts : {}".format(parts))
            if len(parts) == 3:
                return f"{parts[0]}.{parts[1]}.{parts[2]}"
        return date_str
    except Exception as e:
        print(f"[format_date_for_ocr] Erreur : {e}")
        return str(date_input)


@app.post("/verification-globale/")
async def verification_globale(
    donnees: str = Form(...),
    photo_client: UploadFile = File(...),
    photo_id: UploadFile = File(...),
    verso_id: UploadFile = File(...)
):
    try:
        if not all([allowed_file(f.filename) for f in [photo_client, photo_id, verso_id]]):
            return {"status": "error", "message": "Extension de fichiers autorisée : 'jpg, png, jpeg'"}

        user_data = json.loads(donnees)

        # Enregistrements temporaires
        client_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_client.jpg")
        recto_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_recto.jpg")
        verso_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_verso.jpg")

        with open(client_path, "wb") as f:
            shutil.copyfileobj(photo_client.file, f)
        with open(recto_path, "wb") as f:
            shutil.copyfileobj(photo_id.file, f)
        with open(verso_path, "wb") as f:
            shutil.copyfileobj(verso_id.file, f)

        # Prédiction recto/verso
        result = predict_piece_types(recto_path, verso_path)
        
        if result.get("status") != "success":
            return result
        
        recto_type = result["recto_type"]
        recto_conf = result["recto_conf"]
        recto_preds = result["recto_preds"]
        verso_type = result["verso_type"]
        verso_conf = result["verso_conf"]
        verso_preds = result["verso_preds"]
        
        if recto_type != "recto" or recto_conf < 0.5:
            return {
                "status": "error",
                "message": f"Image recto invalide. Confiance recto : {recto_preds['recto'] * 100:.2f}%\n",
                "recto_type": recto_type,
                "recto_conf": recto_conf,
                "recto_preds": recto_preds
            }
        
        if verso_type != "verso" or verso_conf < 0.5:
            return {
                "status": "error",
                "message": f"Image verso invalide. Confiance verso : {verso_preds['verso'] * 100:.2f}%\n",
                "verso_type": verso_type,
                "verso_conf": verso_conf,
                "verso_preds": verso_preds
            }

        # OCR
        ocr_combined = extract_text(recto_path) + "\n" + extract_text(verso_path)
        variance, is_blurry = detect_blur(recto_path)
    
        # Visages
        face_client = detect_face(client_path)
        face_cni = detect_face(recto_path)
    
        if not face_client or not face_cni:
            return {"status": "error", "message": "Visage non détecté"}
    
        face_result = DeepFace.verify(
            face_client,
            face_cni,
            model_name="VGG-Face",
            distance_metric="cosine",
            enforce_detection=False
        )
    
        # Matching informations utilisateur vs OCR
        match_score = 0
        for field in ['nom', 'prenom', 'lieuNaissance']:
            if fuzz.partial_ratio(user_data.get(field, '').lower(), ocr_combined) > 80:
                print("match : {}".format(user_data.get(field, '').lower()))
                match_score += 1
    
        for date_field in ['dateNaissance', 'dateDelivrance', 'dateExpiration', 'numeroPiece']:
            raw_value = user_data.get(date_field, '')
            if not raw_value:
                continue  # Skip si vide ou None
        
            formatted = format_date_for_ocr(raw_value)
            print(f"match : {formatted}")
        
            if fuzz.partial_ratio(formatted, ocr_combined) > 80:
                print(f"match confirmé : {formatted}")
                match_score += 1

 
        print("face verification : {} ".format(face_result['verified']))
        
        
    
        if face_result["verified"] and not is_blurry and match_score >= 4:
           
            message = (
                f"Votre demande est en cours de traitement. Merci de nous faire confiance.\n"
                f"Recto : {recto_preds['recto'] * 100:.2f}% correct\n"
                f"Verso : {verso_preds['verso'] * 100:.2f}% correct."
            )       
   
        
            return {
                "status": "success",
                "ocr_text": ocr_combined,
                "face_verified": True,
                "is_blurry": False,
                "blur_variance": float(variance),
                "match_score": int(match_score),
                "similarity": float(face_result["distance"]),
                "recto_preds": recto_preds,
                "verso_preds": verso_preds,
                "message": message,
                "payload": user_data
            }
        elif not face_result["verified"]:
 
            return {
                "status": "error",
                "message": "Les photos ne correspondent pas (visage différent entre la photo et la pièce d'identité).\n",
                "payload": user_data, 
                "similarity": float(face_result["distance"])
            }

        else :
            return {
                "status": "error",
                "message" : "Les informations fournies sont probablement erronées ou la pièce d'identité est illisible.",
                "payload": user_data,
                "match_score" : match_score, 
                "face_result" : face_result["verified"],
                "ocr_text": ocr_combined,
                }
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

 
