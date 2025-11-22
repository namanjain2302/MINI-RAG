"""Main Gradio application for Mini RAG Assistant (Top-1 retrieval)."""

import gradio as gr
import json
import config
from huggingface_hub import InferenceClient
from utils.document_processor import DocumentProcessor
from utils.embedder import Embedder
from utils.retriever import VectorRetriever
import warnings
warnings.filterwarnings('ignore')


class RAGAssistant:
    """Main RAG Assistant class (Top-1 retrieval)."""

    def __init__(self):
        print("\n" + "="*60)
        print("🚀 Initializing Mini RAG Assistant")
        print("="*60 + "\n")

        # Components
        self.doc_processor = DocumentProcessor()
        self.embedder = Embedder()
        self.retriever = VectorRetriever()

        # Initialize HF text-generation client
        print(f"🔄 Setting up Hugging Face Inference Client for: {config.LLM_MODEL}")
        try:
            self.llm = InferenceClient(
                model=config.LLM_MODEL,
                token=config.HF_API_KEY
            )
            print("✅ Connected to HF Inference API")
        except Exception as e:
            print(f"❌ Could not initialize HF client: {e}")
            self.llm = None

        # Initialize document index if empty
        try:
            if self.retriever.get_collection_count() == 0:
                print("\n📚 No documents indexed. Indexing now...")
                self.index_documents()
        except Exception as e:
            print(f"⚠️ Error checking collection: {e}")

        print("\n" + "="*60)
        print("✅ RAG Assistant Ready!")
        print("="*60 + "\n")

    def index_documents(self):
        documents = self.doc_processor.load_documents()
        if not documents:
            print("⚠ No documents found in data folder.")
            return False

        chunks = self.doc_processor.chunk_documents(documents)
        if not chunks:
            print("⚠ No chunks created.")
            return False

        texts = [c['text'] for c in chunks]
        embeddings = self.embedder.embed_batch(texts)

        self.retriever.add_chunks(chunks, embeddings)
        return True

    def reindex_documents(self):
        print("\n🔄 Re-indexing documents...")
        try:
            self.retriever.reset_collection()
        except:
            pass

        if self.index_documents():
            total = self.retriever.get_collection_count()
            return f"✅ Re-indexed successfully! Total chunks: {total}"
        return "⚠ No documents found."

    def build_prompt(self, query, context):
        return f"""You are a helpful assistant. Use ONLY the context below to answer.

If the answer is not found in the context, reply exactly: "I don't have enough information to answer this question based on the provided documents."

----------------------
CONTEXT:
{context}
----------------------

QUESTION: {query}

ANSWER:"""

    def call_llm(self, prompt):
        """Generate text using HF conversational API."""
        try:
            # Use chat_completion for Llama models (conversational task)
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm.chat_completion(
                messages=messages,
                temperature=0.2,
                max_tokens=300,
                top_p=0.95
            )
            
            # Extract the assistant's reply
            answer = response.choices[0].message.content
            return answer.strip()
            
        except Exception as e:
            return f"❌ LLM Error: {e}"

    def answer_question(self, query: str, history=None) -> str:
        if not query.strip():
            return "Please enter a question."

        try:
            if self.retriever.get_collection_count() == 0:
                return "❌ No documents indexed. Please re-index."
        except:
            pass

        # Retrieval
        q_emb = self.embedder.embed_text(query)
        results = self.retriever.search(q_emb, top_k=1)

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        if not docs:
            return "❌ No relevant context found."

        top_context = docs[0]

        # Build prompt
        prompt = self.build_prompt(query, top_context)

        # LLM answer
        answer = self.call_llm(prompt)

        # Source extraction
        sources = []
        try:
            if isinstance(metas, dict) and "source" in metas:
                sources.append(metas["source"])
            elif isinstance(metas, str):
                sources.append(metas)
        except:
            pass

        if sources:
            answer += f"\n\n📚 Source: {sources[0]}"

        return answer


# ------------------------------------------------------------
# Gradio App
# ------------------------------------------------------------
print("Starting application...")
rag_assistant = RAGAssistant()


def create_interface():
    with gr.Blocks(title="Mini RAG Assistant (Top-1)") as demo:
        gr.Markdown("# 🤖 Mini RAG Assistant (Top-1 Retrieval)")

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat History", height=500)
                msg = gr.Textbox(label="Your Question", placeholder="Ask something...", lines=2)
                with gr.Row():
                    submit = gr.Button("🚀 Submit", variant="primary")
                    clear = gr.Button("🗑 Clear")

            with gr.Column(scale=1):
                gr.Markdown("### 📊 System Info")

                doc_count = gr.Number(
                    label="Indexed Chunks",
                    value=rag_assistant.retriever.get_collection_count(),
                    interactive=False
                )

                reindex_btn = gr.Button("🔄 Re-index Documents")
                reindex_output = gr.Textbox(label="Re-index Status", lines=2)

        # Chat logic - FIXED FOR GRADIO 6.0.0
        def respond(message, chat_history):
            if chat_history is None:
                chat_history = []
            # Add user message as dict with role and content
            chat_history.append({"role": "user", "content": message})
            # Get bot response
            bot_reply = rag_assistant.answer_question(message)
            # Add assistant message as dict with role and content
            chat_history.append({"role": "assistant", "content": bot_reply})
            return "", chat_history

        def do_reindex():
            result = rag_assistant.reindex_documents()
            return result, rag_assistant.retriever.get_collection_count()

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        submit.click(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot)
        reindex_btn.click(do_reindex, None, [reindex_output, doc_count])

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)
