import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"  # Directory containing your book files

def main():
    # Load environment variables
    load_dotenv()
    
    # Verify API key is loaded
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        return
    
    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")
    
    print("Splitting documents into chunks...")
    chunks = split_text(documents)
    print(f"Created {len(chunks)} chunks")
    
    print("Creating embeddings and storing in Chroma...")
    save_to_chroma(chunks)
    print("Database populated successfully!")

def load_documents():
    """Load documents from the data directory"""
    # Load all .md and .txt files from the data/books directory
    loader = DirectoryLoader(DATA_PATH, glob="**/*.md", loader_cls=TextLoader)
    documents = loader.load()
    
    # Also load .txt files if any
    txt_loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader)
    documents.extend(txt_loader.load())
    
    return documents

def split_text(documents):
    """Split documents into smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def save_to_chroma(chunks):
    """Save document chunks to Chroma database"""
    # Clear existing database
    if os.path.exists(CHROMA_PATH):
        import shutil
        shutil.rmtree(CHROMA_PATH)
        print("Cleared existing database")
    
    # Create new database
    embedding_function = OpenAIEmbeddings()
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=CHROMA_PATH
    )
    
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")

if __name__ == "__main__":
    main()