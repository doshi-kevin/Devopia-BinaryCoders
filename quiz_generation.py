import streamlit as st
import tempfile
import re
from transformers import set_seed
set_seed(42)
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
# model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
from langchain_core.prompts import ChatPromptTemplate
quiz_prompt = ChatPromptTemplate.from_template("""You are a teacher preparing questions for a quiz. Given the following document, please generate different four multiple-choice questions (MCQs) with 4 options and a corresponding answer letter based on the context. Do not ask any diagram based questions.

Example question:

Question: question here
CHOICE_A: choice here
CHOICE_B: choice here
CHOICE_C: choice here
CHOICE_D: choice here

Correct Answer:  A or B or C or D. Choice text

Question: question here
CHOICE_A: choice here
CHOICE_B: choice here
CHOICE_C: choice here
CHOICE_D: choice here

Correct Answer:  A or B or C or D. Choice text

These questions should be detailed and solely based on the information provided in the document.

<context>
{context}
<context>""")

from langchain.chains.combine_documents  import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def quiz_generation_tool(documents, embeddings):
    db = FAISS.from_documents(documents, embeddings)
    document_chain_quiz = create_stuff_documents_chain(llm, quiz_prompt)
    retriever = db.as_retriever()
    retrieval_chain_quiz = create_retrieval_chain(retriever, document_chain_quiz)
    return retrieval_chain_quiz

def generate_quiz(retrieval_chain_quiz, example):
    response = retrieval_chain_quiz.invoke({'input': example})
    return response['answer']

def main():
    st.title("Quiz Taking with AI")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(uploaded_file.getvalue())
            temp_path = temp.name
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        documents = text_splitter.split_documents(docs)
        retrieval_chain_quiz = quiz_generation_tool(documents, embeddings)

    input_text = st.text_input("Enter a topic for Quiz:")
    if input_text:
        # Generate quiz questions based on user input
        quiz_questions = generate_quiz(retrieval_chain_quiz, input_text)

        # Display quiz questions
        st.write(quiz_questions)

if __name__ == "__main__":
    main()
