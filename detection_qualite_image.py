import cv2
import numpy as np

# 🛠 Chemin vers l'image (Remplace par ton image)
image_path = "image/doc1.jpg"

# Charger l'image en niveaux de gris
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Vérifier si l'image est bien chargée
if image is None:
    print("❌ Erreur : Impossible de charger l'image ! Vérifie le chemin.")
    exit()

# 1️⃣ Calculer la variance du Laplacien pour détecter le flou
laplacian = cv2.Laplacian(image, cv2.CV_64F)
variance_laplacian = laplacian.var()

print(f"📏 Variance du Laplacien détectée : {variance_laplacian}")

# 2️⃣ L'utilisateur définit son propre seuil pour le flou
seuil_variance = float(input("💡 Entrez le seuil de netteté souhaité (ex : 150) : "))

# 3️⃣ Vérifier si l’image est floue en fonction du seuil donné
if variance_laplacian < seuil_variance:
    print("⚠️ Image floue détectée, amélioration en cours...")

    # Appliquer un filtre de netteté (Unsharp Mask)
    gaussian = cv2.GaussianBlur(image, (9, 9), 10.0)
    sharp_image = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

    # Sauvegarde de l'image améliorée
    cv2.imwrite("image_sharpened.jpg", sharp_image)
    image = sharp_image  # Utiliser l'image améliorée pour l'affichage

    print("✅ Amélioration appliquée !")

else:
    print("✅ L'image est déjà nette, aucune amélioration nécessaire.")

# 📸 Affichage des images
cv2.imshow("Image analysée", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
