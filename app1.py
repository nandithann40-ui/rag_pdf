import os
import streamlit as st
from dotenv import load_dotenv

from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# --------------------------------------------------
# ENV SETUP (ONLY FOR LLM, NOT EMBEDDINGS)
# --------------------------------------------------
load_dotenv()
GOOGLE_API_KEY= os.getenv('GOOGLE_API_KEY')


# --------------------------------------------------
# PDF → TEXT
# --------------------------------------------------
def read_pdfs(pdf_files):
    """
    Extract text from uploaded PDF files
    """
    all_text = ""

    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text

    return all_text


# --------------------------------------------------
# TEXT → CHUNKS
# --------------------------------------------------
def split_into_chunks(text):
    """
    Split large text into overlapping chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    return splitter.split_text(text)


# --------------------------------------------------
# BUILD FAISS INDEX (LOCAL EMBEDDINGS)
# --------------------------------------------------
def build_faiss_index(chunks):
    """
    Create FAISS vector store using local Hugging Face embeddings
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    documents = [Document(page_content=chunk) for chunk in chunks]

    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("faiss_index")


# --------------------------------------------------
# LOAD FAISS INDEX
# --------------------------------------------------
def load_faiss_index():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


# --------------------------------------------------
# PROMPT + LLM
# --------------------------------------------------
def get_prompt_and_llm():
    prompt_template = """
You are an AI assistant.

Answer the question using ONLY the context below.
If the answer is not present in the context, say exactly:
"The answer is not available in the provided context."
The answer has to be in bullet point format, each bullet point has to be in away that a clss 10 th grade student understnd the concept, simplify it as much as possible.
Each bullet point should have maximum 100 words.


Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=1.0,
        google_api_key=GOOGLE_API_KEY
    )

    return prompt, llm


# --------------------------------------------------
# RAG: QUESTION → ANSWER
# --------------------------------------------------
def answer_question(question):
    """
    Complete RAG pipeline:
    Retrieval → Prompt → LLM
    """
    vector_store = load_faiss_index()

    # Retrieve top-k relevant chunks
    docs = vector_store.similarity_search(question, k=10)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt, llm = get_prompt_and_llm()
    final_prompt = prompt.format(
        context=context,
        question=question
    )

    response = llm.invoke(final_prompt)
    return response.content[0]['text']


# --------------------------------------------------
# STREAMLIT APP
# --------------------------------------------------
def main():
    st.set_page_config(page_title="Chat with PDF (Local Embeddings RAG)")

    st.header("📄 Chat with PDF using RAG (Local Embeddings)")

    question = st.text_input("Ask a question from the uploaded PDFs")

    if question:
        try:
            answer = answer_question(question)
            st.subheader("Answer")
            st.write(answer)
        except Exception as exc:
            st.error(f"Unable to generate answer: {exc}")

    with st.sidebar:
        st.title("Upload PDFs")
        pdf_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Process PDFs"):
            if not pdf_files:
                st.warning("Please upload at least one PDF.")
                return

            with st.spinner("Reading PDFs and building vector index..."):
                raw_text = read_pdfs(pdf_files)
                chunks = split_into_chunks(raw_text)

                # Rebuild the index every time so it reflects the newly uploaded PDFs
                build_faiss_index(chunks)

            st.success("PDFs processed and indexed successfully!")


if __name__ == "__main__":
    main()
