import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Détection de voitures")
st.title("🚗 Détection de voitures avec YOLOv8")

# Chargement du modèle (assurez-vous que best.pt est dans le même dossier)
model = YOLO("best.pt")
class_names = model.names  # Pour afficher les noms des classes

# Téléversement de l’image
uploaded_file = st.file_uploader("📤 Téléversez une image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Image téléversée", use_column_width=True)

    image_array = np.array(image)

    with st.spinner("🔍 Détection en cours..."):
        results = model(image_array)
        boxes = results[0].boxes

        if boxes is not None and boxes.cls is not None:
            car_class_index = 2  # Classe "car" dans le dataset COCO
            car_indices = (boxes.cls == car_class_index)
            filtered_boxes = boxes[car_indices]

            if len(filtered_boxes) > 0:
                st.success(f"✅ {len(filtered_boxes)} voiture(s) détectée(s)")

                # Annoter l'image avec les détections
                annotated_img = results[0].plot()
                st.image(annotated_img, caption="📍 Résultat : voitures détectées", use_column_width=True)

                # Afficher les résultats dans un tableau
                st.dataframe(filtered_boxes.data.cpu().numpy())

                # Permettre le téléchargement
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

# Idée UX/UI : Message d’aide
st.markdown("---")
st.info("💡 Astuce : Essayez une image contenant plusieurs voitures pour tester l'efficacité du modèle.")


