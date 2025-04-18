#  YOLOv8 - Détection de Voitures avec Streamlit

Ce projet est une application web simple et interactive développée avec **Python**, **YOLOv8** et **Streamlit**.  
Il permet à un utilisateur de téléverser une image et de détecter automatiquement les **voitures** présentes grâce à un modèle de vision par ordinateur.

---

##  Fonctionnalités

- Téléversement d'images (.jpg, .png, .jpeg)
- Détection **uniquement** des objets de la classe `car` (voiture)
- Affichage :
  - De l’image annotée
  - Du **nombre de voitures détectées**
  - Des **scores de confiance**
  - D’un **tableau de données brut**
- Possibilité de **télécharger l’image annotée**
- Gestion des cas :
  - Aucune voiture détectée
  - Objets non pertinents ignorés

---

##  Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/Khokhane0/MachineLearning.git
cd MachineLearning

### 2. Créer un environnement virtuel (facultatif mais recommandé)

python -m venv env
# Windows
env\Scripts\activate
# macOS/Linux
source env/bin/activate

### 3. Installer les dépendances

pip install -r requirements.txt

### 4. Lancer l'application

streamlit run app.py
L’application s’ouvre automatiquement dans ton navigateur par défaut.

### 5. Exemples d'utilisation

Image test	Résultat attendu
Image avec 3 voitures	Affiche 3 voitures détectées + scores
Image sans voiture	Affiche "Aucune voiture détectée"
Image avec des piétons	Ignore les personnes, ne détecte rien

### 6. Auteur
PAPE KHOKHANE SENE
Contact : via GitHub

### 7. Licence
Ce projet est à visée éducative.
Vous pouvez l’utiliser, le modifier et le distribuer à des fins non commerciales.
