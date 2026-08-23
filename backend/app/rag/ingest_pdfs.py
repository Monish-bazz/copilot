import os
import yaml
import functools
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import RAW_DIR, DATABASE_URL


def _get_db_url():
    db_url = DATABASE_URL
    if db_url and db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+psycopg://", 1)
    elif db_url and db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return db_url


@functools.lru_cache(maxsize=1)
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db_url = _get_db_url()

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="parcelpilot_docs",
        connection=db_url,
        use_jsonb=True,
    )
    return vectorstore


def ingest():
    registry_path = os.path.join(os.path.dirname(__file__), "registry.yaml")
    with open(registry_path, "r") as f:
        registry_data = yaml.safe_load(f)

    docs_to_ingest = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for doc_meta in registry_data.get("documents", []):
        file_name = doc_meta["file"]
        file_path = os.path.join(RAW_DIR, file_name)

        if not os.path.exists(file_path):
            print(f"Warning: File {file_name} not found in {RAW_DIR}")
            continue

        print(f"Processing {file_name}...")

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    chunks = splitter.split_text(text)
                    for chunk in chunks:
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "source_id": doc_meta["source_id"],
                                "title": doc_meta["title"],
                                "status": doc_meta["status"],
                                "authority": doc_meta["authority"],
                                "scope": doc_meta["scope"],
                                "page_number": page_num + 1,
                            },
                        )
                        docs_to_ingest.append(doc)

    print(f"Total chunks created: {len(docs_to_ingest)}")

    if not DATABASE_URL:
        print("DATABASE_URL not set. Skipping vector ingestion.")
        return

    vectorstore = get_vector_store()

    print("Ingesting chunks into pgvector... (this might take a minute)")
    vectorstore.drop_tables()
    vectorstore.create_tables_if_not_exists()
    vectorstore.create_collection()

    vectorstore.add_documents(docs_to_ingest)
    print("PDF ingestion complete.")


if __name__ == "__main__":
    ingest()
