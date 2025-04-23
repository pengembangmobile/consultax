
import streamlit as st
import os
import json
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Membaca API Key dari Streamlit Secrets
os.environ["OPENAI_API_KEY"] = st.secrets["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"]

# Load FAQ dari JSON hasil ebook
with open("ConsultaxAI_EbookPPh2025.json", encoding="utf-8") as f:
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
st.markdown("Tanyakan apa pun tentang **PPh Orang Pribadi**, berbasis Ebook PPh 2025.")

# Input Pertanyaan
query = st.text_input("Pertanyaan Anda:")

if query:
    with st.spinner("Sedang mencari jawaban..."):
        result = qa_chain.invoke({"question": query})
        answer = result.get("answer", "Maaf, saya tidak menemukan jawaban yang relevan.")

        st.markdown("### 💡 Jawaban:")
        st.write(answer)

        # Simpan manual ke memory (hindari ValueError)
        qa_chain.memory.chat_memory.add_user_message(query)
        qa_chain.memory.chat_memory.add_ai_message(answer)
