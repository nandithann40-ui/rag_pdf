import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


load_dotenv()

st.set_page_config(
	page_title="PDF Context",
	page_icon="📄",
	layout="wide",
	initial_sidebar_state="expanded",
)

st.markdown(
	"""
	<style>
	:root { --ink: #1c2421; --muted: #66736d; --accent: #d26a3a; --paper: #f7f3ec; }
	.stApp { background: var(--paper); color: var(--ink); }
	[data-testid="stSidebar"] { background: #e8eee8; border-right: 1px solid #d2ddd3; }
	.hero { padding: 2.5rem 0 1.5rem; border-bottom: 1px solid #d9ded8; }
	.eyebrow { color: var(--accent); font-size: .75rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
	.hero h1 { margin: .35rem 0 .6rem; font-size: clamp(2.2rem, 5vw, 4.5rem); line-height: .95; letter-spacing: 0; }
	.hero p { max-width: 680px; color: var(--muted); font-size: 1.05rem; }
	.status { padding: .8rem 1rem; border-left: 4px solid var(--accent); background: #fffdf8; color: var(--muted); }
	</style>
	""",
	unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the embedding model...")
def create_embeddings() -> HuggingFaceEmbeddings:
	return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def create_llm(api_key: str) -> ChatGoogleGenerativeAI:
	return ChatGoogleGenerativeAI(
		model="models/gemini-3.6-flash",
		temperature=0.3,
		google_api_key=api_key,
	)


@st.cache_data(show_spinner=False)
def extract_text(pdf_files: tuple[tuple[str, bytes], ...]) -> str:
	pages = []
	for _, file_bytes in pdf_files:
		reader = PdfReader(BytesIO(file_bytes))
		pages.extend(page.extract_text() or "" for page in reader.pages)
	return "\n\n".join(page for page in pages if page.strip())


def build_vector_store(text: str) -> FAISS:
	splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
	chunks = splitter.split_text(text)
	if not chunks:
		raise ValueError("No readable text was found in the uploaded PDF.")
	documents = [Document(page_content=chunk) for chunk in chunks]
	return FAISS.from_documents(documents, create_embeddings())


def load_saved_vector_store() -> FAISS:
	return FAISS.load_local(
		"faiss_index",
		create_embeddings(),
		allow_dangerous_deserialization=True,
	)


def response_text(response: object) -> str:
	content = getattr(response, "content", response)
	if isinstance(content, str):
		return content
	if isinstance(content, list):
		return "\n".join(
			item.get("text", "") if isinstance(item, dict) else str(item)
			for item in content
		).strip()
	return str(content)


def answer_question(question: str, vector_store: FAISS, api_key: str) -> str:
	documents = vector_store.similarity_search(question, k=6)
	context = "\n\n".join(document.page_content for document in documents)
	prompt = PromptTemplate.from_template(
		"""You are a helpful AI assistant.
Answer the question using only the context below.
If the answer is not present in the context, say exactly:
THE ANSWER IS NOT AVAILABLE IN THE PROVIDED CONTEXT.
Use concise bullet points. Explain the answer so a class 10 student can understand it.

Context:
{context}

Question:
{question}

Answer:"""
	)
	response = create_llm(api_key).invoke(
		prompt.format(context=context, question=question)
	)
	return response_text(response)


st.markdown(
	"""
	<div class="hero">
	  <div class="eyebrow">PDF context assistant</div>
	  <h1>Ask your document.</h1>
	  <p>Upload a paper, build its local search index, and ask focused questions with answers grounded in its text.</p>
	</div>
	""",
	unsafe_allow_html=True,
)

with st.sidebar:
	st.subheader("Document")
	uploaded_files = st.file_uploader(
		"Upload one or more PDFs",
		type="pdf",
		accept_multiple_files=True,
	)
	if st.button("Build document index", type="primary", use_container_width=True):
		if not uploaded_files:
			st.warning("Upload a PDF first.")
		else:
			file_data = tuple((file.name, file.getvalue()) for file in uploaded_files)
			with st.spinner("Reading and indexing your PDFs..."):
				try:
					text = extract_text(file_data)
					st.session_state.vector_store = build_vector_store(text)
					st.session_state.document_label = ", ".join(name for name, _ in file_data)
					st.success("Document index ready.")
				except Exception as error:
					st.error(f"Could not index the PDFs: {error}")

	st.divider()
	st.caption("The default index is loaded from `faiss_index/` when no upload is active.")

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
	st.error("Add GOOGLE_API_KEY to your .env file before asking a question.")
	st.stop()

if "vector_store" not in st.session_state:
	if os.path.isdir("faiss_index"):
		with st.spinner("Loading the saved document index..."):
			try:
				st.session_state.vector_store = load_saved_vector_store()
				st.session_state.document_label = "attention.pdf"
			except Exception as error:
				st.warning(f"The saved index could not be loaded: {error}")
	else:
		st.session_state.vector_store = None

label = st.session_state.get("document_label", "No document selected")
st.markdown(f'<div class="status"><strong>Source:</strong> {label}</div>', unsafe_allow_html=True)

question = st.chat_input("What would you like to understand?")
if question:
	if st.session_state.get("vector_store") is None:
		st.warning("Upload a PDF and build its index before asking a question.")
	else:
		with st.chat_message("user"):
			st.write(question)
		with st.chat_message("assistant"):
			with st.spinner("Searching the document..."):
				try:
					st.write(answer_question(question, st.session_state.vector_store, api_key))
				except Exception as error:
					st.error(f"The answer could not be generated: {error}")
