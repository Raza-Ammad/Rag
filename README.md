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


