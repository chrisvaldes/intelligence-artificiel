"""import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model("piece_classifier.keras")
LABELS = [ "autre", "recto", "verso"]

def predict_piece_type(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "non_valide", 0.0

    image = cv2.resize(image, (224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    label_index = np.argmax(predictions[0])
    confidence = float(predictions[0][label_index])
    print("Predictions:", predictions[0])
    print("Predicted label:", LABELS[label_index], "with confidence:", confidence)

    return LABELS[label_index], confidence
"""
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model("piece_classifier.keras")
LABELS = ["autre", "recto", "verso"]

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)

def predict_piece_types(recto_path, verso_path):
    recto_image = preprocess_image(recto_path)
    verso_image = preprocess_image(verso_path)

    if recto_image is None or verso_image is None:
        return {"status": "error", "message": "Image(s) invalide(s)."}

    # Prédictions
    recto_preds = model.predict(recto_image)[0]
    verso_preds = model.predict(verso_image)[0]

    # Index et confiance
    recto_index = np.argmax(recto_preds)
    verso_index = np.argmax(verso_preds)

    recto_type = LABELS[recto_index]
    verso_type = LABELS[verso_index]

    recto_conf = float(recto_preds[recto_index])
    verso_conf = float(verso_preds[verso_index])

    return {
        "status": "success",
        "recto_type": recto_type,
        "recto_conf": recto_conf,
        "recto_preds": {label: float(p) for label, p in zip(LABELS, recto_preds)},
        "verso_type": verso_type,
        "verso_conf": verso_conf,
        "verso_preds": {label: float(p) for label, p in zip(LABELS, verso_preds)},
        "valid": recto_type == "recto" and recto_conf >= 0.8 and verso_type == "verso" and verso_conf >= 0.8
    }
