# 🔍 CBIR YOLOv8 Project - Content-Based Image Retrieval

## 📋 Description

Système de recherche d'images par contenu utilisant YOLOv8 pour la détection d'objets et des descripteurs de features pour la similarité.

## 🎯 Fonctionnalités

1. **Détection d'objets** : YOLOv8 détecte automatiquement les objets dans les images
2. **Extraction de features** : Calcul de descripteurs (forme, couleur, texture, contours)
3. **Recherche par similarité** : Trouve des objets similaires dans la base de données
4. **Interface web** : Upload, analyse, recherche et gestion d'images

## 🏗️ Architecture

```
Frontend (HTML/CSS/JS) ←→ API Flask ←→ YOLOv8 + Descripteurs
```

## 📁 Structure du projet

```
cbir-yolo-project/
├── backend/          # API Flask (Nayrouz)
├── frontend/         # Interface web (Salma/Aya)
├── notebooks/        # Notebooks Colab (Nayrouz)
├── datasets/         # Images d'entraînement
└── docs/            # Documentation
```

## 🚀 Installation

### Backend (API)

```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend

```bash
cd frontend
# Ouvrir index.html dans un navigateur
```

## 📊 Workflow

1. **Upload** : L'utilisateur upload une image
2. **Détection** : YOLO détecte les objets
3. **Extraction** : Calcul des descripteurs pour chaque objet
4. **Recherche** : Comparaison avec la base de données
5. **Résultats** : Affichage des images similaires

## 🛠️ Technologies

- **Backend** : Flask, YOLOv8, OpenCV, NumPy
- **Frontend** : HTML5, CSS3, JavaScript
- **ML** : Ultralytics YOLOv8, scikit-image
- **Storage** : JSON (features database)

## 📝 Documentation

- [Tâches de l'équipe](docs/tasks.md)
- [Documentation API](docs/api_documentation.md)
- [Explication du projet](docs/project_explanation.md)

## 🔗 API Endpoints

- `POST /api/detect` : Détecte les objets dans une image
- `POST /api/extract` : Extrait les descripteurs
- `POST /api/search` : Cherche des objets similaires
- `GET /api/images` : Liste toutes les images
- `DELETE /api/images/<id>` : Supprime une image

## 📈 Statut du projet

- [x] Structure du projet
- [x] Fonctions de descripteurs
- [ ] Entraînement YOLO (Aya)
- [ ] API Flask (Nayrouz)
- [ ] Interface web (Salma)
- [ ] Intégration complète
- [ ] Tests finaux

## 📅 Timeline

- **Semaine 1** : Préparation (descripteurs + structure)
- **Semaine 2** : Intégration YOLO + API
- **Semaine 3** : Frontend + Tests

## 🤝 Contribution

1. Clone le repo
2. Crée une branche : `git checkout -b feature/ma-fonctionnalite`
3. Commit : `git commit -m "Ajout de ma fonctionnalité"`
4. Push : `git push origin feature/ma-fonctionnalite`
5. Crée une Pull Request

## 📧 Contact

Pour toute question, contactez l'équipe sur le groupe WhatsApp/Discord.

---

**Version** : 1.0.0  

**Dernière mise à jour** : Décembre 2025
