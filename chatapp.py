import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult, Generation
from dotenv import load_dotenv
import requests
import json
from typing import List, Optional, Any

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Custom embeddings class to make SentenceTransformer compatible with LangChain
class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)

# Custom LLM class for Groq API
class GroqLLM(BaseLLM):
    api_key: str
    url: str = "https://api.groq.com/openai/v1/chat/completions"
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.3
    max_tokens: int = 1000

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        self.api_key = api_key

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> LLMResult:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompts[0]}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        if stop:
            payload["stop"] = stop

        response = requests.post(self.url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        generations = [
            [Generation(text=result["choices"][0]["message"]["content"])]
        ]
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "groq"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    if not text_chunks:
        st.error("No text chunks to process. Please upload valid PDF files.")
        return
    embeddings = SentenceTransformerEmbeddings('all-MiniLM-L6-v2')
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    st.success("Vector store created and saved.")

def get_conversational_chain():
    if not GROQ_API_KEY:
        st.error("Groq API key is missing. Please set the GROQ_API_KEY in the .env file.")
        return None

    prompt_template = """
    You are a helpful assistant. Answer the question as detailed as possible based on the provided context. If the answer is not in the context, say, "Answer is not available in the context." Do not provide incorrect information.

    Context: {context}

    Question: {question}

    Answer:
    """

    model = GroqLLM(api_key=GROQ_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = SentenceTransformerEmbeddings('all-MiniLM-L6-v2')
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Failed to load FAISS index: {str(e)}. Please process the PDF files first.")
        return

    docs = new_db.similarity_search(user_question)
    if not docs:
        st.warning("No relevant documents found for the query.")
        return

    chain = get_conversational_chain()
    if not chain:
        return

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    # Display the reply in a styled box without copy button
    reply_text = response["output_text"].replace('"', '"').replace('\n', '<br>')  # Escape quotes and preserve line breaks
    st.markdown(f"""
        <div class="response-box">
            <div class="response-text">Reply: {reply_text}</div>
        </div>
    """, unsafe_allow_html=True)

def main():
    # Set page configuration with a generic title and icon
    st.set_page_config(page_title="PDF Query Assistant", page_icon="📚", layout="wide")

    # Custom CSS for a modern, interactive look with visible text
    st.markdown("""
        <style>
        .main {
            background-color: #f5f7fa;
            padding: 20px;
            border-radius: 10px;
        }
        .stButton > button {
            background-color: #007bff;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #0056b3;
            transform: scale(1.05);
        }
        .stTextInput > div > div > input {
            border: 2px solid #007bff;
            border-radius: 8px;
            padding: 10px;
        }
        .stFileUploader > div > div {
            border: 2px dashed #007bff;
            border-radius: 8px;
            padding: 20px;
            background-color: #e9ecef;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
            border-right: 1px solid #dee2e6;
            padding: 20px;
            border-radius: 10px;
        }
        .stAlert {
            border-radius: 8px;
        }
        h1, h2, h3 {
            color: #343a40;
            font-family: 'Arial', sans-serif;
        }
        .query-box {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .response-box {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .response-text {
            color: #1a1a1a !important;  /* Dark color for visibility */
            font-size: 16px;
            line-height: 1.5;
        }
        .response-text p, .response-text div {
            color: #1a1a1a !important;  /* Ensure all nested elements are dark */
        }
        </style>
    """, unsafe_allow_html=True)

    # Main content area
    st.title("📚 PDF Query Assistant")
    st.markdown("Ask questions about your uploaded PDF documents and get precise answers powered by AI.")

    # Query input section
    with st.container():
        st.markdown('<div class="query-box">', unsafe_allow_html=True)
        st.subheader("Enter Your Question")
        user_question = st.text_input(
            "Type your question here...",
            placeholder="e.g., What is the main topic of the document?",
            key="user_question"
        )
        if user_question:
            with st.spinner("Fetching answer..."):
                user_input(user_question)
        st.markdown('</div>', unsafe_allow_html=True)

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.header("📁 Upload PDFs")
        st.markdown("Upload one or more PDF files to query their content.")
        pdf_docs = st.file_uploader(
            "Choose PDF files",
            accept_multiple_files=True,
            type=["pdf"],
            help="Select one or more PDF files to process."
        )
        SECONDARY_BUTTON = "secondary"
        st.button("Process PDFs", key="process_button", type=SECONDARY_BUTTON)
        if st.session_state.get("process_button"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text:
                        st.error("No text extracted from the PDF files. Please check the files.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
        st.markdown("---")
        st.markdown("**Instructions:**\n1. Upload your PDF files.\n2. Click 'Process PDFs' to analyze them.\n3. Ask your question in the main panel.")

if __name__ == "__main__":
    main()