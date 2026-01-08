## RAG QA conversation with PDF including chat history
import os
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")

## set up streamlit app
st.title("Conversational RAG with PDF and Chat History")
st.write("Upload PDF's and chat with their content.")

## input groq api key
groq_api_key = st.text_input("Enter your Groq API Key", type="password")

##check if api key is provided
if groq_api_key:
    llm=ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")
    
    ## chat interface
    session_id=st.text_input("Session ID", value="default_session")
    ## statefully manage chat history
    if "store" not in st.session_state:
        st.session_state.store={}
    
    uploaded_file=st.file_uploader("Upload a PDF file", type="pdf",accept_multiple_files=True)
    ## process uploaded PDF
    if uploaded_file:
        document=[]
        for file in uploaded_file:
            temppdf=f"./temp.pdf"
            with open(temppdf, "wb") as f:
                f.write(file.getvalue())

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            document.extend(docs)
        
        # split and create embeddings for the documents
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits=text_splitter.split_documents(document)
        vectorstore=Chroma.from_documents(documents=splits,embedding=embeddings)
        retrieval=vectorstore.as_retriever()
    
        contextualize_q_system_prompt=(
            "Given a chat history and latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history.Do not answer the question, "
            "just reformulate it if needed otherwise return the question as is."
        )
        contextualize_q_system_prompt= ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chathistory"),
                ("human", "{input}"),
            ]
        )
        
        history_aware_retriever=create_history_aware_retriever(llm,retrieval,contextualize_q_system_prompt)
        
        ## Answer Question prompt
        system_prompt=(
            "You are an assistant for question answering. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "use three sentences maximum to answer the question. "
            "Keep the answer concise."
            "\n\n"
            "{context}"
        )
        
        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chathistory"),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chathistory",
            output_messages_key="answer"
        )
        
        user_input=st.text_input("Your question:")
        if user_input:
            session_history=get_session_history(session_id)
            response=conversational_rag_chain.invoke(
                {"input":user_input},
                config={"configurable":{"session_id":session_id}}
            )
            st.write(st.session_state.store)
            st.success(f"Assistant: {response["answer"]}")
            st.write("chat history:",session_history.messages)
else:
    st.warning("Please enter your Groq API Key to use the application.")
