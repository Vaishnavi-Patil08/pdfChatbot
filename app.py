import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from typing import List
from dotenv import load_dotenv
load_dotenv()

class sentence_transformer(Embeddings):
    def __init__(self,model_name):
        self.model=SentenceTransformer(model_name)
    def embed_documents(self, texts:List[str])->List[List[float]]:
        return self.model.encode(texts).tolist()
    def embed_query(self, text:str)->List[float]:
        return self.model.encode([text])[0].tolist()

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("ChatPDF")

uploaded = st.file_uploader("Upload PDF", type="pdf")
if uploaded:
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_file.write(uploaded.getbuffer())
    tmp_file.flush()
    tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = splitter.split_documents(docs)


    embedding_model = sentence_transformer(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(
        documents=final_docs,
        embedding=embedding_model,
    )
    retriever = db.as_retriever()

    prompt=ChatPromptTemplate([("""Answer the following question based only on the provided context. Think step by step before providing a detailed answer.
                          <context>
                          {context}
                          </context>
                          Question: {input} """),])
    
    llm= ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key= os.getenv("GOOGLE_API_KEY"))

    doc_chain=create_stuff_documents_chain(prompt=prompt,llm=llm)

    retrieval_chain=create_retrieval_chain(retriever,doc_chain)

    query = st.text_input("Ask a question about your pdf")
    if query:
        res = retrieval_chain.invoke({"input": query})
        st.write("**Answer:**", res["answer"]) 

else:
    st.info("Please upload a PDF to get started.")
