 


import cv2
import numpy as np

def detect_blur(image_path, threshold=300):
    """
    Détecte si une image est floue en utilisant la variance du Laplacien.
    
    :param image_path: Chemin de l'image
    :param threshold: Seuil pour déterminer si l'image est floue (300 par défaut)
    :return: (variance, is_blurry) -> variance du Laplacien et True si l'image est floue
    """
    # Charger l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Erreur : Impossible de charger l'image.")
        return None, None

    # Appliquer l'opérateur de Laplacien
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Calculer la variance du Laplacien
    variance = laplacian.var()

    # Déterminer si l'image est floue
    is_blurry = variance < threshold

    # Afficher le résultat
    print(f"Variance du Laplacien : {variance:.2f}")
    if is_blurry:
        print("⚠️ L'image est floue !")
    else:
        print("✅ L'image est nette !")

    return variance, is_blurry

# Exemple d'utilisation
image_path = "moi1.jpeg"  # Remplace par le chemin de ton image
detect_blur(image_path)













