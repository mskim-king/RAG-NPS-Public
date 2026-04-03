# vector_store.py
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore

CHROMA_PATH = "./chroma_db"
CACHE_PATH = "./cache"
COLLECTION_NAME = "rag_collection"

def build_vector_store(file_path: str) -> str:
    # load
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    recursive_docs = text_splitter.split_documents(docs)

    # Embed
    underlying_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    store = LocalFileStore(CACHE_PATH)

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings,
        store,
        namespace=underlying_embeddings.model
    )

    # Store
    Chroma.from_documents(
        documents=recursive_docs,
        embedding=cached_embedder,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print('벡터스토어 생성 완료')

    return "벡터스토어 생성 완료"

def load_vector_store():
    underlying_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    store = LocalFileStore(CACHE_PATH)

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings,
        store,
        namespace=underlying_embeddings.model
    )

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=cached_embedder,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space" : "cosine"},
    )

    return db

def get_retriever(k: int = 2):
    db = load_vector_store()
    retriever = db.as_retriever(search_kwargs={"k": k})
    return retriever