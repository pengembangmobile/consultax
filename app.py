
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

# Load konten JSON
with open("ConsultaxAI_EbookPPh2025_deskripsi.json", encoding="utf-8") as f:
    raw_data = json.load(f)

# Buat dokumen
docs = [
    Document(
        page_content=item["content"],
        metadata={
            "title": item.get("title", ""),
            "category": item.get("category", ""),
            "source": item.get("source", "")
        }
    )
    for item in raw_data
    if item.get("content")
]

# Buat vectorstore
embedding = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embedding)
retriever = db.as_retriever()

# Buat memory lokal
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Buat chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model_name="gpt-4"),
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)

# Streamlit UI
st.set_page_config(page_title="ConsultaxAI (GPT-4)", page_icon="💬")
st.title("🤖 ConsultaxAI – Konsultan Pajak AI (Berbasis Ebook PPh 2025)")
st.markdown("Tanyakan apa pun tentang **PPh Orang Pribadi**, berdasarkan isi konten Ebook PPh 2025.")

# Input Pertanyaan
query = st.text_input("Pertanyaan Anda:")

if query:
    with st.spinner("Sedang mencari jawaban..."):
        response = qa_chain.invoke({"input": query})  # fix: gunakan "input" key
        answer = response.get("answer", "Maaf, tidak menemukan jawaban yang relevan.")
        sources = response.get("source_documents", [])

        st.markdown("### 💡 Jawaban:")
        st.write(answer)

        if sources:
            st.markdown("#### 📚 Sumber:")
            for i, doc in enumerate(sources):
                title = doc.metadata.get("title", "Tidak diketahui")
                source = doc.metadata.get("source", "")
                st.markdown(f"- **{title}** – *{source}*")

        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(answer)
