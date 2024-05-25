import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from git import Repo, GitCommandError
from dotenv import load_dotenv
import time

load_dotenv()

# Load the API keys from the environment
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.title("GitHub Repository Q&A")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

class Document:
    def __init__(self, page_content, file_path):
        self.page_content = page_content
        self.metadata = {"file_path": file_path}

def clone_github_repo(repo_url, repo_path="./repo", retries=3):
    for attempt in range(retries):
        try:
            if os.path.exists(repo_path):
                os.system(f"rm -rf {repo_path}")
                # pass
            Repo.clone_from(repo_url, repo_path)
            return
        except GitCommandError as e:
            if attempt < retries - 1:
                st.warning(f"Retrying clone... ({attempt + 1}/{retries})")
                time.sleep(2)
            else:
                st.error(f"Failed to clone the repository after {retries} attempts. Error: {str(e)}")
                raise e

def load_code_files(repo_path="./repo"):
    code_files = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith((".py", ".java", ".js", ".ts", ".cpp", ".c", ".h", ".cs", ".md", ".txt","ipynb")):
                with open(os.path.join(root, file), "r") as f:
                    code_files.append(Document(page_content=f.read(), file_path=os.path.join(root, file)))
    return code_files

def vector_embedding():
    repo_url = st.session_state.repo_url
    repo_path = "./repo"

    clone_github_repo(repo_url, repo_path)  # Clone the GitHub repository
    st.session_state.docs = load_code_files(repo_path)  # Load code files

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # st.session_state.embeddings = OpenAIEmbeddings()  # Change to a compatible embeddings model
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Splitting
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector OpenAI embeddings

# Initialize session state variables if they don't exist
if 'repo_url' not in st.session_state:
    st.session_state.repo_url = ""
if 'docs' not in st.session_state:
    st.session_state.docs = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'text_splitter' not in st.session_state:
    st.session_state.text_splitter = None
if 'final_documents' not in st.session_state:
    st.session_state.final_documents = None
if 'vectors' not in st.session_state:
    st.session_state.vectors = None

prompt1 = st.text_input("Enter Your Question From Repository")

repo_url = st.text_input("Enter GitHub Repository URL")
if st.button("Load Repository"):
    st.session_state.repo_url = repo_url
    try:
        vector_embedding()
        st.write("Vector Store DB Is Ready")
    except GitCommandError:
        st.error("Failed to load the repository. Please check the URL or try again later.")

if prompt1 and st.session_state.vectors is not None:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write(f"Response time: {time.process_time() - start:.2f} seconds")
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
