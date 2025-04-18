import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from io import BytesIO

# 🎯 Configuration de la page
st.set_page_config(page_title="Détection de voitures - YOLOv8", layout="centered")

st.title("🚗 Détection de voitures avec YOLOv8")
st.markdown("**Téléversez une image pour détecter automatiquement les voitures.**")

# 🔁 Chargement du modèle
model = YOLO("best.pt")
class_names = model.names  # dict : {0: 'person', 1: 'bicycle', 2: 'car', ...}

# 📤 Téléversement de l’image
uploaded_file = st.file_uploader("📷 Choisir une image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 📸 Chargement et affichage
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((640, 640))  # 🛠️ Redimension pour meilleure perf
    st.image(image, caption="Image téléversée", use_column_width=True)

    image_array = np.array(image)

    with st.spinner("🔍 Détection en cours..."):
        results = model(image_array)
        boxes = results[0].boxes
        car_class_index = 2  # Classe "car" dans le dataset COCO

        if boxes is not None and boxes.cls is not None:
            car_indices = boxes.cls == car_class_index
            filtered_boxes = boxes[car_indices]

            if len(filtered_boxes) > 0:
                st.success(f"✅ {len(filtered_boxes)} voiture(s) détectée(s).")

                # 📈 Afficher les scores de confiance pour chaque voiture détectée
                for box in filtered_boxes:
                    conf = float(box.conf[0])
                    st.write(f"Voiture détectée avec {conf:.2%} de confiance.")

                # 📍 Image annotée
                annotated_img = results[0].plot()
                st.image(annotated_img, caption="📌 Résultat : voitures détectées", use_column_width=True)

                # 📊 Tableau brut des résultats
                st.dataframe(filtered_boxes.data.cpu().numpy())

                # 💾 Bouton de téléchargement de l’image annotée
                output_image = Image.fromarray(annotated_img)
                buffer = BytesIO()
                output_image.save(buffer, format="PNG")
                st.download_button(
                    label="📥 Télécharger l'image annotée",
                    data=buffer.getvalue(),
                    file_name="voitures_detectees.png",
                    mime="image/png"
                )
            else:
                st.warning("🚫 Aucune voiture détectée dans cette image.")
        else:
            st.error("❌ Aucun objet détecté ou erreur dans l’image.")

# 💡 UX : Astuce à l'utilisateur
st.markdown("---")
st.info("💡 Essayez une image contenant plusieurs voitures pour tester la précision du modèle.")

# 🎨 UX personnelle : thème clair/sombre suggestion
st.markdown("🖌️ Amélioration UX : Ajouter un sélecteur de thème (clair/sombre) pourrait rendre l'application plus accessible.")

# ⏱️ Si modèle lent : redimension de l'image + modèle léger
st.markdown("⚙️ Optimisation : Pour accélérer la détection, l'image a été redimensionnée à 640x640 et le modèle léger `yolov8n.pt` est recommandé.")
