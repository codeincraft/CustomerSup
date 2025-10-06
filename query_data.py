import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Path for the local vector store
CHROMA_PATH = "chroma"

# Template for the model
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Streamlit Page Setup
st.set_page_config(
    page_title="LangChain Q&A with Streamlit",
    page_icon="🤖",
    layout="centered"
)

# App Title
st.title("📚 LangChain Q&A with Chroma & OpenAI")
st.write("Ask questions based on your embedded documents using **LangChain** and **Chroma**.")

# Load API key from Streamlit secrets (works both locally and on cloud)
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = api_key
except Exception as e:
    st.error("❌ OPENAI_API_KEY not found in secrets. Please configure it in Streamlit Cloud settings.")
    st.stop()

# Initialize Embedding and Database
@st.cache_resource(show_spinner=False)
def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return db

db = load_vectorstore()

# Input from user
user_query = st.text_input("💬 Ask a question from your document:", placeholder="e.g., What is the summary of chapter 3?")

# Process Query
if st.button("🔍 Generate Answer"):
    if not user_query.strip():
        st.warning("Please enter a question before submitting.")
        st.stop()
    
    with st.spinner("Searching and generating answer... ⏳"):
        results = db.similarity_search_with_relevance_scores(user_query, k=3)
        
        if not results:
            st.error("⚠️ No matching results found in the vector database.")
            st.stop()
        
        best_score = results[0][1]
        if best_score < 0.5:
            st.warning(f"Low confidence in search results (Best score: {best_score:.2f})")
            st.stop()
        
        # Create context from results
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=user_query)
        
        # Call the OpenAI model
        model = ChatOpenAI(temperature=0.3)
        response = model.predict(prompt)
    
    # Display Answer
    st.success("✅ **Answer:**")
    st.write(response)
    
    # Show Sources
    sources = [doc.metadata.get("source", "Unknown") for doc, _ in results]
    st.subheader("📎 Sources")
    st.write(", ".join(sources))
    
    # Debug Info (Expandable)
    with st.expander("🔍 Debug Info"):
        st.write("**Similarity Scores:**")
        for i, (doc, score) in enumerate(results):
            st.write(f"**Result {i+1}:** Score = {score:.4f}")
            st.write(doc.page_content[:300] + "...")
            st.write("---")