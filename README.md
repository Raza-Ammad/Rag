# RAG PDF Chat – Streamlit + Gemini + Pinecone

This project is a **Retrieval-Augmented Generation (RAG)** app that lets you:

- Upload PDFs
- Ask natural language questions about them
- See which document chunks were used as context
- Get answers powered by **Google Gemini**

It’s built to be:

- Simple to run locally
- Easy to demo to others
- Easy to deploy to **Streamlit Community Cloud**

---

## Table of Contents

- [RAG PDF Chat – Streamlit + Gemini + Pinecone](#rag-pdf-chat--streamlit--gemini--pinecone)
  - [Table of Contents](#table-of-contents)
  - [✨ Key Features](#-key-features)
  - [🏗️ System Architecture](#️-system-architecture)
    - [Current (Local + Streamlit Cloud) Architecture](#current-local--streamlit-cloud-architecture)
    - [Future (More Advanced) Architecture Ideas](#future-more-advanced-architecture-ideas)
  - [🔒 Security Notice](#-security-notice)
  - [🔧 Prerequisites](#-prerequisites)
  - [🚀 Deployment Option 1: Run Locally (Recommended for Demos)](#-deployment-option-1-run-locally-recommended-for-demos)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Create and Activate a Virtual Environment](#2-create-and-activate-a-virtual-environment)
    - [3. Install Dependencies](#3-install-dependencies)
    - [4. Set Environment Variables (API Keys)](#4-set-environment-variables-api-keys)
    - [5. Add Documents to `docs/`](#5-add-documents-to-docs)
    - [6. Ingest Documents into Pinecone](#6-ingest-documents-into-pinecone)
    - [7. Run the Streamlit App](#7-run-the-streamlit-app)
  - [🚀 Deployment Option 2: Deploy to Streamlit Community Cloud](#-deployment-option-2-deploy-to-streamlit-community-cloud)
  - [⚙️ Configuration](#️-configuration)
  - [💡 How the App Works (RAG Flow)](#-how-the-app-works-rag-flow)
  - [💻 Local Development Workflow](#-local-development-workflow)
  - [🔍 Troubleshooting Guide](#-troubleshooting-guide)
  - [❓ Frequently Asked Questions (FAQ)](#-frequently-asked-questions-faq)
  - [🛣️ Future Improvements](#️-future-improvements)

---

## ✨ Key Features

| Feature                         | Description                                                                                                   |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **PDF & TXT support**           | Ingests `.pdf` and `.txt` files and makes them queryable via semantic search.                                |
| **Chat-style interface**        | Streamlit chat UI to ask natural language questions and see answers like a conversational assistant.         |
| **Gemini-powered answers**      | Uses Google **Gemini (gemini-flash-latest)** to generate grounded answers from retrieved context.            |
| **Pinecone vector store**       | Stores document embeddings in a **Pinecone** serverless index, suitable for cloud deployment and scaling.    |
| **Context transparency**        | Shows which chunks were retrieved from which document so you can verify where the answer came from.         |
| **PDF upload in the UI**        | Upload new PDFs directly from the app and index them into Pinecone on the fly.                               |
| **Environment-based config**    | Uses environment variables locally and **Streamlit secrets** in the cloud.                                   |
| **Easy local setup**            | Single `requirements.txt`, one ingestion script, and a single `app.py` entry point.                          |

---

## 🏗️ System Architecture

### Current (Local + Streamlit Cloud) Architecture

The current setup is designed for:

- **Local demos** on a laptop
- **Hosted demos** via Streamlit Community Cloud

```text
  [User Browser]
        │
        │ 1. Ask question / upload PDF
        ▼
┌───────────────────────────┐
│       Streamlit App       │
│         (app.py)          │
└───────────────────────────┘
        │ ▲
        │ │ 2. Embed query / chunks
        ▼ │
┌───────────────────────────┐
│ SentenceTransformer Model │
│  (all-MiniLM-L6-v2)       │
└───────────────────────────┘
        │
        │ 3. Query / upsert vectors
        ▼
┌───────────────────────────┐
│    Pinecone Index         │
│   (serverless, "big")     │
└───────────────────────────┘
        │
        │ 4. Context (chunks + source)
        ▼
┌───────────────────────────┐
│   Gemini API (AI Studio)  │
│   gemini-flash-latest     │
└───────────────────────────┘
        │
        │ 5. Answer
        ▼
┌───────────────────────────┐
│       Streamlit App       │
│  (shows answer + context) │
└───────────────────────────┘
```
🔧 Prerequisites

To run this app, you’ll need:
	•	Python 3.11+
	•	A Google AI Studio account and a Gemini API key
	•	A Pinecone account and:
	•	A serverless index (e.g. named big)
	•	(Optional) A Streamlit Community Cloud account if you want a hosted demo

🚀 Deployment Option 1: Run Locally
```

1. 	Clone the Repository
		git clone https://github.com/Raza-Ammad/Rag.git
		cd Rag
	
2. 	Create and Activate a Virtual Environment
   	macOS / Linux:
   	python3 -m venv venv
		source venv/bin/activate
		Windows (PowerShell):
		python -m venv venv
		venv\Scripts\Activate.ps1

3. 	Install Dependencies
   	pip install -r requirements.txt

4. 	Set Environment Variables (API Keys)
		You need three environment variables:
		•	GOOGLE_API_KEY – Gemini key from Google AI Studio
		•	PINECONE_API_KEY – key from Pinecone dashboard
		•	PINECONE_INDEX_NAME – name of your Pinecone index 

5. 	Add Documents to docs/
   	docs/
		├── example1.txt
		└── Tunnel Vision.pdf

6. 	Ingest Documents into Pinecone
   	python ingest_pinecone.py

7. 	Run the Streamlit App
   	streamlit run app.py
```
🚀 Deployment Option 2: Deploy to Streamlit Community Cloud
```
If you want a hosted demo:

1.	Make the GitHub repo public (Settings → Danger Zone → Change visibility → Public).
2.	Go to https://share.streamlit.io and log in with GitHub.
3.	Click “New app” and select:
	•	Repository: Raza-Ammad/Rag (or your fork)
	•	Branch: main
	•	Main file: app.py
4.	In app → Settings → Secrets, add:
		GOOGLE_API_KEY = "YOUR_GEMINI_KEY"
		PINECONE_API_KEY = "YOUR_PINECONE_KEY"
		PINECONE_INDEX_NAME = "big"
5.	Save and deploy.
```
⚙️ Configuration
```
Most configuration is handled via environment variables (locally) or Streamlit secrets (in the cloud).
Name										Where Used					Description
GOOGLE_API_KEY					app & ingest				Gemini API key

PINECONE_API_KEY				app & ingest				Pinecone API key

PINECONE_INDEX_NAME			app & ingest				Pinecone index name

Paths / constants:
	•	docs/ – folder with base documents (.txt / .pdf)
	•	app.py – main Streamlit RAG UI
	•	ingest_pinecone.py – batch ingestion script for docs/

```
💡 How the App Works (RAG Flow)

At a high level
	1.	Documents are read, chunked, embedded, and stored in Pinecone.
	2.	A user asks a question in the Streamlit app.
	3.	The question is embedded and sent to Pinecone to retrieve relevant chunks.
	4.	The app builds a prompt with the retrieved chunks + the user question.
	5.	The prompt is sent to Gemini, which produces an answer.
	6.	The app shows:
	•	The answer
	•	The chunks used (source + text)
	•	The chat history for the session

	sequenceDiagram
    participant U as User
    participant S as Streamlit app
    participant P as Pinecone
    participant G as Gemini

    U->>S: Ask question
    S->>P: Embed query & search top_k
    P-->>S: Return chunks (text + source)
    S->>G: Build prompt (context + question)
    G-->>S: Answer based on context
    S-->>U: Show answer + retrieved context




