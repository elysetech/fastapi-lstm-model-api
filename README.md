# API FastAPI pour Déploiement de Modèle LSTM

Une application FastAPI complète pour déployer et servir des prédictions à partir d'un modèle LSTM (Long Short-Term Memory).

*[English version below](#lstm-model-api)*

## Table des matières

- [API FastAPI pour Déploiement de Modèle LSTM](#api-fastapi-pour-déploiement-de-modèle-lstm)
  - [Table des matières](#table-des-matières)
  - [Aperçu](#aperçu)
  - [Fonctionnalités](#fonctionnalités)
  - [Architecture](#architecture)
  - [Prérequis](#prérequis)
  - [Installation](#installation)
  - [Exécution de l'API](#exécution-de-lapi)
    - [Méthode 1: Scripts de démarrage](#méthode-1-scripts-de-démarrage)
    - [Méthode 2: Ligne de commande](#méthode-2-ligne-de-commande)
    - [Méthode 3: Docker](#méthode-3-docker)
  - [Documentation de l'API](#documentation-de-lapi)
  - [Points d'Accès (Endpoints)](#points-daccès-endpoints)
    - [Vérification de Santé](#vérification-de-santé)
    - [Prédiction](#prédiction)
    - [Prédiction à partir d'un Fichier](#prédiction-à-partir-dun-fichier)
    - [Prédiction par Lots](#prédiction-par-lots)
    - [Informations sur le Modèle](#informations-sur-le-modèle)
  - [Utilisation de Votre Propre Modèle LSTM](#utilisation-de-votre-propre-modèle-lstm)
  - [Exemples d'Utilisation](#exemples-dutilisation)
    - [Client Python](#client-python)
    - [Utilisation avec cURL](#utilisation-avec-curl)
    - [Script d'Exemple](#script-dexemple)
  - [Déploiement](#déploiement)
    - [Docker](#docker)
    - [Cloud](#cloud)
  - [Tests](#tests)
  - [Structure du Projet](#structure-du-projet)
  - [Licence](#licence)
- [LSTM Model API](#lstm-model-api)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Architecture](#architecture-1)
  - [Requirements](#requirements)
  - [Installation](#installation-1)
  - [Running the API](#running-the-api)
    - [Method 1: Startup Scripts](#method-1-startup-scripts)
    - [Method 2: Command Line](#method-2-command-line)

## Aperçu

Ce projet fournit une API RESTful pour effectuer des prédictions avec un modèle LSTM pré-entraîné. L'API est construite avec FastAPI, offrant une documentation automatique, une validation des données et des performances élevées.

## Fonctionnalités

- **Prédictions en temps réel** via plusieurs endpoints:
  - Prédiction à partir de données JSON
  - Prédiction à partir de fichiers uploadés
  - Prédictions par lots (batch)
- **Documentation interactive** générée automatiquement
- **Validation des données** d'entrée et de sortie
- **Gestion des erreurs** robuste
- **Conteneurisation Docker** pour un déploiement facile
- **Scripts de démarrage** pour Windows et Unix/Linux
- **Tests automatisés** pour vérifier la fonctionnalité de l'API
- **Exemple de client** pour démontrer l'utilisation de l'API

## Architecture

L'application est structurée selon les principes de conception modernes:

```
├── main.py           # Application FastAPI principale
├── model.py          # Implémentation du modèle LSTM
├── example.py        # Client exemple
├── test_api.py       # Tests automatisés
├── requirements.txt  # Dépendances
├── Dockerfile        # Configuration Docker
├── docker-compose.yml # Configuration Docker Compose
├── run.sh            # Script de démarrage pour Unix/Linux
└── run.bat           # Script de démarrage pour Windows
```

## Prérequis

- Python 3.8+
- pip (gestionnaire de paquets Python)
- Virtualenv (recommandé)
- Docker (optionnel, pour le déploiement conteneurisé)

## Installation

1. Clonez le dépôt:

```bash
git clone <url-du-dépôt>
cd api-lstm-fastapi
```

2. Créez un environnement virtuel (recommandé):

```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. Installez les dépendances:

```bash
pip install -r requirements.txt
```

4. Placez votre modèle LSTM pré-entraîné dans le répertoire du projet (optionnel):
   - Si vous avez un modèle pré-entraîné, sauvegardez-le sous `model/lstm_model.h5`
   - Si aucun modèle n'est fourni, un modèle simplifié sera créé pour démonstration

## Exécution de l'API

### Méthode 1: Scripts de démarrage

Utilisez les scripts fournis pour démarrer l'API:

```bash
# Sur Unix/Linux/Mac
./run.sh

# Sur Windows
run.bat
```

### Méthode 2: Ligne de commande

Démarrez le serveur API manuellement:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Méthode 3: Docker

Utilisez Docker pour exécuter l'API dans un conteneur:

```bash
# Construire l'image
docker build -t lstm-api .

# Exécuter le conteneur
docker run -p 8000:8000 lstm-api

# Ou avec Docker Compose
docker-compose up
```

L'API sera disponible à l'adresse http://localhost:8000

## Documentation de l'API

Une fois le serveur démarré, vous pouvez accéder à la documentation interactive:

- Interface Swagger: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Points d'Accès (Endpoints)

### Vérification de Santé

```
GET /
```

Retourne un message simple pour confirmer que l'API fonctionne.

**Réponse:**

```json
{
  "status": "ok",
  "message": "LSTM Model API is running"
}
```

### Prédiction

```
POST /predict
```

Effectue une prédiction avec le modèle LSTM en utilisant des données JSON.

**Corps de la Requête:**

```json
{
  "sequence": [[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.6], ...],
  "sequence_length": 10
}
```

**Réponse:**

```json
{
  "prediction": [0.75],
  "confidence": 0.95
}
```

### Prédiction à partir d'un Fichier

```
POST /predict/file
```

Téléchargez un fichier JSON contenant des données de séquence pour la prédiction.

**Requête:**
- Données de formulaire avec un champ de téléchargement de fichier nommé "file"

**Réponse:**

```json
{
  "prediction": [0.75],
  "confidence": 0.95
}
```

### Prédiction par Lots

```
POST /batch-predict
```

Effectue des prédictions pour plusieurs séquences d'entrée.

**Corps de la Requête:**

```json
[
  {
    "sequence": [[0.1, 0.2, 0.3, 0.4, 0.5], ...],
    "sequence_length": 10
  },
  {
    "sequence": [[0.2, 0.3, 0.4, 0.5, 0.6], ...],
    "sequence_length": 10
  }
]
```

**Réponse:**

```json
[
  {
    "prediction": [0.75],
    "confidence": 0.95
  },
  {
    "prediction": [0.82],
    "confidence": 0.92
  }
]
```

### Informations sur le Modèle

```
GET /model-info
```

Obtient des informations sur le modèle LSTM chargé.

**Réponse:**

```json
{
  "model_type": "LSTM (Simplified)",
  "input_shape": [10, 5],
  "output_shape": [1],
  "layers": [
    {
      "name": "input_layer",
      "type": "Input",
      "units": null,
      "activation": null
    },
    {
      "name": "lstm_layer",
      "type": "SimplifiedLSTM",
      "units": 8,
      "activation": "tanh"
    },
    {
      "name": "output_layer",
      "type": "Dense",
      "units": 1,
      "activation": "linear"
    }
  ],
  "total_params": 129,
  "is_sample_model": true,
  "note": "This is a simplified model for demonstration purposes only"
}
```

## Utilisation de Votre Propre Modèle LSTM

Pour utiliser votre propre modèle LSTM pré-entraîné:

1. Sauvegardez votre modèle TensorFlow/Keras en utilisant `model.save('chemin/vers/modele.h5')`
2. Modifiez le fichier `model.py` pour charger votre modèle
3. Si votre modèle nécessite un prétraitement spécifique, mettez à jour la méthode `preprocess` dans `model.py`
4. Assurez-vous que les dépendances TensorFlow sont décommentées dans `requirements.txt`

## Exemples d'Utilisation

### Client Python

```python
import requests
import numpy as np

# Préparer les données d'entrée
sequence = np.random.rand(10, 5).tolist()  # 10 pas de temps, 5 caractéristiques

# Faire une prédiction
response = requests.post(
    "http://localhost:8000/predict",
    json={"sequence": sequence}
)

# Afficher le résultat
print(response.json())
```

### Utilisation avec cURL

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"sequence": [[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.6]]}'
```

### Script d'Exemple

Le projet inclut un script `example.py` qui démontre comment:
- Se connecter à l'API
- Obtenir des informations sur le modèle
- Générer des données d'exemple
- Faire des prédictions
- Visualiser les résultats

Pour exécuter l'exemple:

```bash
python example.py
```

## Déploiement

### Docker

Le projet inclut un `Dockerfile` et un fichier `docker-compose.yml` pour faciliter le déploiement:

```bash
# Construire et démarrer avec Docker Compose
docker-compose up -d

# Ou construire manuellement
docker build -t lstm-api .
docker run -p 8000:8000 lstm-api
```

### Cloud

Pour déployer sur des plateformes cloud:

1. **AWS Elastic Beanstalk**:
   - Utilisez le Dockerfile fourni
   - Créez un environnement Docker sur Elastic Beanstalk
   - Déployez votre application

2. **Google Cloud Run**:
   - Construisez l'image Docker
   - Poussez-la vers Google Container Registry
   - Déployez sur Cloud Run

3. **Azure App Service**:
   - Utilisez le support Docker d'App Service
   - Déployez en utilisant le Dockerfile fourni

## Tests

Le projet inclut des tests automatisés pour vérifier la fonctionnalité de l'API:

```bash
python test_api.py
```

Les tests vérifient:
- La disponibilité de l'API
- Les informations du modèle
- Les prédictions simples
- Les prédictions par lots
- Les prédictions à partir de fichiers
- La gestion des erreurs

## Structure du Projet

```
├── main.py           # Application FastAPI principale
├── model.py          # Implémentation du modèle LSTM
├── example.py        # Client exemple
├── test_api.py       # Tests automatisés
├── requirements.txt  # Dépendances
├── Dockerfile        # Configuration Docker
├── docker-compose.yml # Configuration Docker Compose
├── .dockerignore     # Fichiers à ignorer dans le build Docker
├── .gitignore        # Fichiers à ignorer dans Git
├── run.sh            # Script de démarrage pour Unix/Linux
└── run.bat           # Script de démarrage pour Windows
```

## Licence

[MIT License](LICENSE)

---

# LSTM Model API

A complete FastAPI application for deploying and serving predictions from an LSTM (Long Short-Term Memory) model.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the API](#running-the-api)
- [API Documentation](#api-documentation)
- [API Endpoints](#api-endpoints)
- [Using Your Own LSTM Model](#using-your-own-lstm-model)
- [Example Usage](#example-usage)
- [Deployment](#deployment)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [License](#license)

## Overview

This project provides a RESTful API for making predictions with a pre-trained LSTM model. The API is built with FastAPI, providing automatic documentation, validation, and high performance.

## Features

- **Real-time predictions** via multiple endpoints:
  - Prediction from JSON data
  - Prediction from uploaded files
  - Batch predictions
- **Interactive documentation** automatically generated
- **Input and output data validation**
- **Robust error handling**
- **Docker containerization** for easy deployment
- **Startup scripts** for Windows and Unix/Linux
- **Automated tests** to verify API functionality
- **Example client** to demonstrate API usage

## Architecture

The application is structured according to modern design principles:

```
├── main.py           # Main FastAPI application
├── model.py          # LSTM model implementation
├── example.py        # Example client
├── test_api.py       # Automated tests
├── requirements.txt  # Dependencies
├── Dockerfile        # Docker configuration
├── docker-compose.yml # Docker Compose configuration
├── run.sh            # Startup script for Unix/Linux
└── run.bat           # Startup script for Windows
```

## Requirements

- Python 3.8+
- pip (Python package manager)
- Virtualenv (recommended)
- Docker (optional, for containerized deployment)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd lstm-api-fastapi
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Place your trained LSTM model in the project directory (optional):
   - If you have a pre-trained model, save it as `model/lstm_model.h5`
   - If no model is provided, a simplified model will be created for demonstration

## Running the API

### Method 1: Startup Scripts

Use the provided scripts to start the API:

```bash
# On Unix/Linux/Mac
./run.sh

# On Windows
run.bat
```

### Method 2: Command Line

