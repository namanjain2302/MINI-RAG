# Mini RAG-Powered Assistant
A modular Retrieval-Augmented Generation chatbot using open-source LLMs, Gradio UI, Hugging Face deployment, and Chroma vector DB for document-based Q&A from custom knowledge sources.


## Features
- Ingest and chunk custom documents (PDF, TXT).
- Embedding via Hugging Face models.
- Store/retrieve document vectors with Chroma.
- Query handling and natural answers with LLM.
- User-friendly Gradio web interface.
- One-click deployment to Hugging Face Spaces.


## Tech Stack
- **Front-End:** Gradio (UI)
- **Embeddings:** Hugging Face Transformers (sentence-transformers)
- **Vector Database:** Chroma DB (vector store)
- **Generative Component:** Open-source llm (Meta-llama 3.1) 
- Python 3.10+

## Workflow (Data Flow)
<img src="rag.png" alt="RAG Architecture" width="700" />
