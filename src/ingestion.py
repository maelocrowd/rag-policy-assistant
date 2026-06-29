# ingestion.py

from pathlib import Path
import uuid

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredHTMLLoader,
)
from utils import (
    CORPUS_DIR,
    CHROMA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    get_embeddings,
)


def load_documents(directory_path: str):
    """
    Loads supported document types.

    Supported formats:
        - Markdown (.md, .markdown)
        - Text (.txt)
        - PDF (.pdf)
        - HTML (.html, .htm)
    """

    documents = []

    supported_patterns = [
        "**/*.md",
        "**/*.markdown",
        "**/*.txt",
        "**/*.pdf",
        "**/*.html",
        "**/*.htm",
    ]

    for pattern in supported_patterns:

        for file_path in Path(directory_path).glob(pattern):

            print(f"Processing: {file_path}")

            suffix = file_path.suffix.lower()

            try:

                if suffix in [".md", ".markdown"]:

                    text = file_path.read_text(
                        encoding="utf-8"
                    )

                    documents.append(
                        {
                            "content": text,
                            "source": file_path.name,
                            "path": str(file_path),
                            "type": "markdown",
                        }
                    )

                elif suffix == ".txt":

                    loader = TextLoader(
                        str(file_path),
                        encoding="utf-8",
                    )

                    for doc in loader.load():

                        documents.append(
                            {
                                "content": doc.page_content,
                                "source": file_path.name,
                                "path": str(file_path),
                                "type": "text",
                            }
                        )

                elif suffix == ".pdf":

                    loader = PyPDFLoader(str(file_path))

                    pages = loader.load()

                    documents.append(
                        {
                            "content": "\n".join(
                                page.page_content
                                for page in pages
                            ),
                            "source": file_path.name,
                            "path": str(file_path),
                            "type": "pdf",
                        }
                    )

                elif suffix in [".html", ".htm"]:

                    loader = UnstructuredHTMLLoader(
                        str(file_path)
                    )

                    docs = loader.load()

                    documents.append(
                        {
                            "content": docs[0].page_content,
                            "source": file_path.name,
                            "path": str(file_path),
                            "type": "html",
                        }
                    )

            except Exception as e:

                print(f"Skipping {file_path}: {e}")

    return documents

def chunk_documents(documents):
    """
    Chunks loaded documents.

    Markdown preserves heading hierarchy.

    Other document types use recursive chunking.
    """

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ],
        strip_headers=False,
    )

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = []

    for document in documents:

        if document["type"] == "markdown":

            header_chunks = markdown_splitter.split_text(
                document["content"]
            )

            for chunk in header_chunks:

                chunk.metadata["source"] = document["source"]
                chunk.metadata["file_path"] = document["path"]
                chunk.metadata["document_type"] = document["type"]
                chunk.metadata["chunk_id"] = str(uuid.uuid4())
                chunk.metadata["chunk_length"] = len(chunk.page_content)
                if len(chunk.page_content) > CHUNK_SIZE:

                    chunks.extend(
                        recursive_splitter.split_documents(
                            [chunk]
                        )
                    )

                else:

                    chunks.append(chunk)

        else:

            

            doc = Document(
                page_content=document["content"],
                metadata={
                    "source": document["source"],
                    "file_path": document["path"],
                    "document_type": document["type"],
                },
            )

            chunks.extend(
                recursive_splitter.split_documents([doc])
            )

    return chunks

def build_vector_store(chunks):
    """
    Creates embeddings and stores them
    in the Chroma vector database.
    """

    print("Loading embedding model...")

    embeddings = get_embeddings()

    print(
        f"Initializing and storing vectors inside "
        f"{CHROMA_DIR}..."
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print("Vector database created successfully.")


def main():

    corpus_path = Path(CORPUS_DIR)

    if not corpus_path.exists():
        print(
            f"Error: Target directory "
            f"'{CORPUS_DIR}' does not exist."
        )
        return

    print("Beginning document chunking strategy...")

    documents = load_documents(CORPUS_DIR)

    chunks = chunk_documents(documents)

    if not chunks:
        print("No documents were processed.")
        return

    print(f"Generated {len(chunks)} total text chunks.")

    build_vector_store(chunks)

    print("Ingestion execution completed successfully!")


if __name__ == "__main__":
    main()