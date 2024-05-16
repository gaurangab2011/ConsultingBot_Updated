import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
import asyncio
from autogen import AssistantAgent, UserProxyAgent
import openpyxl

#Upload PDF file

st.header("My first chatbot")


class TrackableAssistantAgent(AssistantAgent):
    def _process_received_message(self, message, sender, silent):
        with st.chat_message(sender.name):
            st.markdown(message)
        return super()._process_received_message(message, sender, silent)


class TrackableUserProxyAgent(UserProxyAgent):
    def _process_received_message(self, message, sender, silent):
        with st.chat_message(sender.name):
            st.markdown(message)
        return super()._process_received_message(message, sender, silent)


with st.sidebar:
    st.header("OpenAI Configuration")
    selected_model = st.selectbox("Model", ['gpt-3.5-turbo', 'gpt-4'], index=1)
    selected_key = st.text_input("API Key", type="password")

with st.sidebar:
    st.title("Your documents")
    file = st.file_uploader("Upload a PDF file and start asking your questions", type="pdf")

#Extract the text

if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        #st.write(text)

    #Break it into chunks

    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len

    )
    chunks = text_splitter.split_text(text)
    #st.write(chunks)

    # Generate embeddings

    embeddings = OpenAIEmbeddings(openai_api_key=selected_key)

    # Create vector db

    vector_store = FAISS.from_texts(chunks, embeddings)

    user_question = st.text_input("Type Your question here")

    # do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
        # st.write(match)

        # define the LLM
        llm = ChatOpenAI(
            openai_api_key=selected_key,
            temperature=0,
            max_tokens=1000,
            model_name=selected_model
        )

        # output results
        # chain -> take the question, get relevant document, pass it to the LLM, generate the output
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_question)
        st.write(response)

        # file_path = "C:/Users/Gauranga/Desktop/Consulting sample.xlsx"
        # workbook = openpyxl.load_workbook(file_path)
        # sheet = workbook.active
        # sheet['B2'] = response

        llm_config = {
            "timeout": 600,
            "seed": 42,
            "config_list": [
                {
                    "model": selected_model,
                    "api_key": selected_key
                }
            ]
        }
        # create an AssistantAgent instance named "assistant"
        assistant = TrackableAssistantAgent(
            name="assistant", llm_config=llm_config, code_execution_config={"use_docker": False})

        # create a UserProxyAgent instance named "user"
        user_proxy = TrackableUserProxyAgent(
            name="user", human_input_mode="NEVER", llm_config=llm_config)

        # Create an event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


        # Define an asynchronous function
        async def initiate_chat():
            await user_proxy.a_initiate_chat(
                assistant,
                message=response,
            )


        # Run the asynchronous function within the event loop
        loop.run_until_complete(initiate_chat())
