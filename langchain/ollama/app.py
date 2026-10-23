import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM


load_dotenv()

## LangSmith tracing
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "ollama-chatbot")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://aws.api.smith.langchain.com")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked."),
        ("user", "Question: {question}"),
    ]
)

st.title("This is my langchain demo. where i'm using LLAMA2 model")
input_text = st.text_input("What question you have in mind.")

llm = OllamaLLM(model='llama3')
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))


# ## Prompt template
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant. Please respond to the user's question."),
#         ("user", "Question: {question}"),
#     ]
# )


# def generate_response(question, model_name, temperature):
#     llm = OllamaLLM(model=model_name, temperature=temperature)
#     chain = prompt | llm | StrOutputParser()
#     return chain.invoke({"question": question})


# ## Streamlit UI
# st.title("Q&A Chatbot with Ollama")

# st.sidebar.title("Settings")
# model_name = st.sidebar.selectbox("Select model", ["llama3", "gemma:2b"])
# temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)

# if not os.environ["LANGSMITH_API_KEY"]:
#     st.sidebar.warning("LANGSMITH_API_KEY not set — tracing is disabled.")
# else:
#     st.sidebar.success(f"Tracing to LangSmith project: {os.environ['LANGSMITH_PROJECT']}")

# question = st.text_input("Ask a question")

# if question:
#     with st.spinner("Thinking..."):
#         st.write(generate_response(question, model_name, temperature))
# else:
#     st.write("Please enter a question.")
