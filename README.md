# 🎬 Video Chatbot

Chatbot qui crée des vidéos (et leur voix off) à partir d'une conversation, avec ou sans images existantes.

- **Front** : Streamlit (liste des dernières conversations, lecteurs vidéo/audio)
- **Back** : FastAPI
- **Orchestration** : LangGraph (un orchestrateur choisit l'outil : chat, vidéo, audio, statut)
- **Mémoire** : historique des conversations dans SQLite
- **Modèles** : tout passe par **OpenRouter** (une seule clé API pour le LLM, la vidéo et la voix)

## Fonctionnalités

- Conversation avec mémoire, plusieurs conversations sauvegardées (sidebar)
- Génération vidéo à partir d'un texte
- Génération vidéo à partir d'images existantes : animer une image, ou l'utiliser comme référence
- Voix off (synthèse vocale), seule ou enchaînée après la vidéo
- Récupération différée d'une vidéo (« où en est ma vidéo ? »)
- Mode mock pour tester tout le flux sans payer de génération

## Architecture

```
video-chatbot/
├── backend/
│   ├── main.py         # API FastAPI (conversations, chat, médias)
│   ├── graph.py        # Graphe LangGraph : orchestrateur + nœuds
│   ├── db.py           # Persistance SQLite (conversations + messages)
│   ├── video_tool.py   # OpenRouter vidéo : submit -> poll -> download
│   ├── audio_tool.py   # OpenRouter voix off (TTS)
│   └── media.py        # Stockage local des fichiers générés (/media)
├── frontend/
│   └── app.py          # Interface Streamlit
├── requirements.txt
└── .env.example
```

### Le graphe LangGraph

```
START → orchestrator ─┬→ chat ─────────────────────────→ END
                      ├→ generate_audio ───────────────→ END
                      ├→ video_status ─────────────────→ END
                      └→ generate_video ─┬─(avec voix)─→ generate_audio → END
                                         └─(sans)──────→ END
```

L'orchestrateur (un LLM en sortie structurée) choisit la route, reformule le prompt vidéo et
extrait le texte de la voix off. À chaque message, `main.py` recharge l'historique depuis SQLite,
le passe au graphe, puis enregistre les nouveaux messages.

## Prérequis

- Python 3.10+
- Une clé API [OpenRouter](https://openrouter.ai)
- Streamlit ≥ 1.43 (pour joindre des images dans la barre de saisie)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows : copy .env.example .env
```

Renseigne au minimum `OPENROUTER_API_KEY` dans `.env`.

## Configuration (`.env`)

| Variable | Rôle | Défaut / exemple |
|---|---|---|
| `OPENROUTER_API_KEY` | Clé unique pour LLM, vidéo et voix | obligatoire |
| `LLM_MODEL` | LLM de l'orchestrateur et du chat (doit gérer les appels de fonctions) | `anthropic/claude-sonnet-4.5` |
| `VIDEO_MODEL` | Modèle de génération vidéo | `heygen/heygen-video-1` |
| `VIDEO_RESOLUTION` | Résolution (valeurs valides selon le modèle) | vide = défaut du modèle |
| `VIDEO_ASPECT_RATIO` | Format d'image (ex. `16:9`) | vide = défaut du modèle |
| `VIDEO_GENERATE_AUDIO` | `1` / `0`, envoyé seulement s'il est défini | non défini |
| `TTS_MODEL` | Modèle de synthèse vocale | à renseigner pour la voix off réelle |
| `TTS_VOICE` | Voix du TTS | `alloy` |
| `MOCK_MEDIA` | `1` = vidéo/audio d'exemple, aucun appel payant | `1` |
| `DB_PATH` | Fichier SQLite | `chatbot.db` |
| `MEDIA_DIR` | Dossier des médias générés | `media` |
| `PUBLIC_BASE_URL` | URL publique du back (pour les liens médias) | `http://localhost:8000` |
| `API_URL` | URL du back utilisée par Streamlit | `http://localhost:8000` |
| `MAX_IMAGE_MB` | Taille max d'une image jointe | `10` |

Les paramètres vidéo (durée, résolution, format, images de début/fin) sont vérifiés d'après
`https://openrouter.ai/api/v1/videos/models` : un paramètre non supporté par le modèle n'est jamais envoyé.

## Lancer

Deux terminaux, depuis le dossier du projet :

```bash
# Terminal 1 : back
cd backend
uvicorn main:app --reload

# Terminal 2 : front
cd frontend
streamlit run app.py
```

- Interface : http://localhost:8501
- API et documentation interactive : http://localhost:8000/docs

## Utilisation

Exemples de messages :

- `Fais une vidéo d'un chat sur une plage au coucher du soleil`
- `Fais une voix off en français qui dit : « Bienvenue sur ma chaîne »`
- `Fais une vidéo de montagne avec une voix off qui dit : « Bonjour à tous »`
- `Où en est ma vidéo ?`

**Avec des images** : clique sur 📎 dans la barre de saisie, puis choisis dans la sidebar :

- *Animer l'image* : la 1re image devient la première image de la vidéo. Une 2e image (image de fin) n'est utilisée que si le modèle la supporte.
- *Image(s) de référence* : les images guident le style ou le contenu (souvent facturé plus cher).

Dans les deux cas, décris le **mouvement** voulu (« zoom lent, vent léger »), pas l'image elle-même.

### Tester sans rien payer

Avec `MOCK_MEDIA=1`, la vidéo et la voix sont des fichiers d'exemple. Seul le LLM est appelé :
la clé OpenRouter et `LLM_MODEL` sont donc nécessaires.

## API

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/conversations` | Dernières conversations |
| `POST` | `/conversations` | Crée une conversation |
| `GET` | `/conversations/{id}/messages` | Historique |
| `POST` | `/conversations/{id}/chat` | Envoie un message (formulaire : `message`, `image_mode`, `images`) |
| `DELETE` | `/conversations/{id}` | Supprime une conversation |
| `GET` | `/media/{fichier}` | Vidéos, audios et images stockés |

```bash
curl -X POST http://localhost:8000/conversations
curl -X POST http://localhost:8000/conversations/<ID>/chat \
  -F "message=Anime cette image, zoom lent" \
  -F "image_mode=first_frame" \
  -F "images=@photo.jpg"
```

## Modèles et coûts

La génération vidéo est **facturée à la seconde**, selon le modèle et la résolution. Vérifie les tarifs
sur la page du modèle dans OpenRouter : ils changent (promotions comprises).

Exemple avec `heygen/heygen-video-1` (capacités lues sur `/videos/models`) :

- résolutions : 480p, 768p, 2K ; durées : 5 à 15 s par clip
- image de début uniquement (pas d'image de fin)
- tarif de liste : 0,02 / 0,03 / 0,09 $ par seconde (480p / 768p / 2K), doublé en mode « référence »

Pour tester à petit prix : un clip de 5 s en 480p, avant de lancer des clips plus longs.
Pour une vidéo longue, il faut générer plusieurs clips (15 s maximum chacun) puis les assembler.

## Dépannage

| Symptôme | Cause probable |
|---|---|
| `Connection refused` dans Streamlit | Le back n'est pas lancé, ou pas sur le port 8000 |
| Erreur 401 / 403 | `OPENROUTER_API_KEY` absente ou mal copiée |
| Erreur au démarrage mentionnant une autre clé (ex. `DEEPSEEK_API_KEY`) | Le LLM est créé avec `init_chat_model` au lieu du client OpenRouter : utilise la version de `graph.py` fournie |
| `ImportError` / `NameError` à l'import | Fichiers de versions différentes dans `backend/` : garde un jeu cohérent |
| Une demande vidéo reçoit une réponse de chat | Le routage dépend du LLM : formule explicitement (« génère une vidéo de… ») ou change `LLM_MODEL` |
| Erreur 400 sur la vidéo | Paramètre non supporté (durée, résolution…) : le message de l'API s'affiche dans le chat |
| « La génération audio a échoué » | `TTS_MODEL` vide ou invalide |
| Vidéo « en cours » | Normal : 30 s à plusieurs minutes, redemande « où en est ma vidéo ? » |

## Limites connues et pistes

- Un clip dure quelques secondes : pas d'assemblage automatique de plusieurs clips ni de fusion vidéo + voix off (piste : nœud `ffmpeg`).
- La musique n'est pas branchée (seule la voix off l'est).
- Les images ne servent que pour le message où elles sont jointes.
- Authentification et multi-utilisateurs absents : projet pensé pour un usage local.
- SQLite : lancer uvicorn avec un seul worker.
