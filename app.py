import streamlit as st
# from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from cssTemplates import css,user_template,bot_template
from langchain_community.llms import Ollama
# from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
import joblib
from langchain.schema import Document


def get_raw_data(pdfs):
    text=""
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_chunks(data):
    splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(data)
    return chunks

def create_vector_store(chunks):
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        show_progress=True
    )
    vectorstore = FAISS.from_texts(texts=chunks,embedding=embeddings)
    faiss.write_index(vectorstore.index,'vectorstore_saved.index')
    docstore_dict = {k: {'page_content': v.page_content, 'metadata': v.metadata} for k, v in vectorstore.docstore._dict.items()}
    joblib.dump(docstore_dict, 'docstore.pkl')
    joblib.dump(vectorstore.index_to_docstore_id, 'index_to_docstore_id.pkl')
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = Ollama(model="qwen2:1.5b")

    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True
    )
    retriever = vectorstore.as_retriever()
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_input):
    response = st.session_state.conversation({'question': user_input})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="chat with multiple pdf",page_icon=":books:")
    st.write(css,unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("chat with multiple pdfs :books:")
    user_input = st.text_input("Ask a question")

    with st.sidebar:
        st.subheader("upload your pdfs")
        pdfs = st.file_uploader("choose your files",accept_multiple_files=True)

        
        if st.button("Process"):
            # if pdfs:
                with st.spinner("Processing"):
                    # print("done")
                    # text_data = get_raw_data(pdfs)

                    # chunks = get_chunks(text_data)

                    # vectorstore = create_vector_store(chunks)

                    index = faiss.read_index('vectorstore_saved.index')
                    docstore_dict = joblib.load('docstore.pkl')
                    index_to_docstore_id = joblib.load('index_to_docstore_id.pkl')
                    docstore = InMemoryDocstore({k: Document(page_content=v['page_content'], metadata=v['metadata']) for k, v in docstore_dict.items()})
                    embeddings = OllamaEmbeddings(
                        model="nomic-embed-text",
                        show_progress=True
                    )
                    vectorstore = FAISS(embedding_function=embeddings, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
            # else:
            #     st.sidebar.write("Please Upload files first!!!")

    if user_input:
        if st.session_state.conversation==None:
            st.write(bot_template.replace(
                "{{MSG}}", "Please upload files first!!!"), unsafe_allow_html=True)
        else:
            handle_user_input(user_input)

if __name__ == "__main__":
    main()