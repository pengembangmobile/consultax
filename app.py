
import streamlit as st
import os
import json
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Membaca API Key dari Streamlit Secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Load FAQ dari JSON
with open("ConsultaxAI_FAQ_PPh_Orang_Pribadi_130.json", encoding="utf-8") as f:
    faq_data = json.load(f)

docs = [
    Document(
        page_content=f"Q: {item['question']}\nA: {item['answer']}",
        metadata={
            "category": item.get("category", ""),
            "source": item.get("source", "")
        }
    )
    for item in faq_data
]

# Buat vectorstore
embedding = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embedding)
retriever = db.as_retriever()

# Buat memory untuk percakapan
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Inisiasi QA Chain berbasis memory
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model_name="gpt-4"),
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)

# Streamlit UI
st.set_page_config(page_title="ConsultaxAI (GPT-4)", page_icon="💬")
st.title("🤖 ConsultaxAI – Konsultan Pajak AI (GPT-4 + Conversational Memory)")
st.markdown("Tanyakan apa pun tentang **PPh Orang Pribadi**, berbasis FAQ dan regulasi perpajakan.")

# Input Pertanyaan
query = st.text_input("Pertanyaan Anda:")

if query:
    with st.spinner("Sedang mencari jawaban..."):
        result = qa_chain.invoke({"question": query})
        answer = result.get("answer", "Maaf, saya tidak menemukan jawaban yang relevan.")
        
        st.markdown("### 💡 Jawaban:")
        st.write(answer)
