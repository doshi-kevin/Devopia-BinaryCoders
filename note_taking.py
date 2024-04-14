import streamlit as st
import tempfile
import os
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Initialize your models and loaders here
llm = HuggingFaceEndpoint(repo_id="google/gemma-1.1-7b-it", huggingfacehub_api_token='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', temperature=0.3)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def summarization_tool(documents, embeddings):
    db = FAISS.from_documents(documents, embeddings)
    prompt = ChatPromptTemplate.from_template("""Provide a summary of the {input} based on provided context.
    Provide answer in detail to help student learn.
    <context>
    {context}
    <context>""")
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever=db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def response_summarization(retrieval_chain, example):
    response = retrieval_chain.invoke({'input': example})
    return response['answer']

def main():
    st.title("Note Making with AI")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(uploaded_file.getvalue())
            temp_path = temp.name
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        documents = text_splitter.split_documents(docs)
        retrieval_chain = summarization_tool(documents, embeddings)

        input_text = st.text_input("Enter a topic for summarization:")
        if input_text:
            # Split input by delimiter (e.g., comma) and process each line
            input_lines = input_text.split(',')
            for line in input_lines:
                line = line.strip() # Remove leading/trailing whitespace
                if line: # Check if line is not empty
                    summary = response_summarization(retrieval_chain, line)
                    st.write(f"Summary for '{line}': {summary}")
        os.unlink(temp_path)

if __name__ == "__main__":
    main()
