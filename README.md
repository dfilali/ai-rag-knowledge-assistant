# AG Knowledge Assistant - Chatbot IA sur Documents (RAG)

Ce projet est un assistant intelligent basé sur le patron de conception **RAG (Retrieval-Augmented Generation)**. Il permet d'importer des documents PDF, de les découper en morceaux (chunks), de calculer leurs embeddings vectoriels, de les indexer localement dans une base **FAISS**, puis de dialoguer avec eux de façon fluide grâce à une mémoire conversationnelle intégrée.

L'application propose également un système d'**auto-évaluation** pour mesurer la fidélité de la réponse (détection d'hallucinations) et la pertinence de l'assistant par rapport à la question.

---

## 🚀 Fonctionnalités Clés

1. **Importation & Indexation PDF** : Découpage intelligent par page avec conservation des métadonnées sources (nom du fichier, numéro de page).
2. **Double Moteur LLM** : Support d'**OpenAI (GPT-4o-mini)** et de **Mistral AI (Mistral Large)**, sélectionnable dynamiquement dans l'interface.
3. **Mémoire de Conversation** : Conservation du contexte des échanges précédents. Le chatbot reformule automatiquement les questions de suivi pour garantir des recherches vectorielles précises.
4. **Sources Transparentes** : Affichage interactif des citations exactes utilisées par l'IA pour générer sa réponse.
5. **Auto-Évaluation RAG** : Évaluation automatisée par LLM de la fidélité (*faithfulness*) et de la pertinence (*relevance*) des réponses, notée sur 5 étoiles avec des explications détaillées.
6. **Interface Premium** : Design moderne sur une seule page (SPA) avec thème sombre, effets de flou (glassmorphism) et micro-animations.

---

## 🛠️ Stack Technique

- **Backend** : FastAPI (Python)
- **Framework IA** : LangChain (langchain-core, langchain-community)
- **Base Vectorielle** : FAISS-CPU (indexation locale rapide sans serveur externe)
- **Modèles de Langage** : OpenAI (embeddings + chat) / Mistral AI (embeddings + chat)
- **Frontend** : HTML5, Vanilla CSS (Design sur-mesure), Vanilla JavaScript

---

## ⚙️ Installation et Lancement

### Option 1 : Lancement en local (Python)

#### 1. Cloner et entrer dans le dossier du projet
```bash
git clone https://github.com/votre-compte/ai-rag-knowledge-assistant.git
cd ai-rag-knowledge-assistant
```

#### 2. Créer et activer l'environnement virtuel
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Installer les dépendances backend
```bash
pip install -r backend/requirements.txt
```

#### 4. Configurer les clés API
Copiez le fichier `.env.example` en `.env` :
```bash
cp .env.example .env
```
Éditez le fichier `.env` pour ajouter vos clés :
```env
OPENAI_API_KEY=votre_cle_openai
MISTRAL_API_KEY=votre_cle_mistral
```
*(Note : Si vous ne configurez pas le fichier `.env`, vous pourrez toujours entrer vos clés directement dans le panneau de configuration de l'interface graphique).*

#### 5. Lancer le serveur de développement
```bash
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```
Ouvrez votre navigateur sur [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

### Option 2 : Lancement avec Docker (Recommandé)

Le projet est entièrement conteneurisé. Docker gère toutes les dépendances système complexes (notamment les dépendances OpenMP pour FAISS).

#### 1. Lancer l'application
Assurez-vous que Docker est démarré, puis exécutez la commande suivante à la racine du projet :
```bash
docker compose up --build
```

#### 2. Accéder à l'interface
Une fois le conteneur démarré, l'application est accessible à l'adresse suivante :
[http://localhost:8000](http://localhost:8000)

Les PDF téléversés et les index vectoriels sont sauvegardés sur votre machine hôte dans le répertoire `./data`, ce qui évite de perdre vos documents lors de l'arrêt du conteneur.
