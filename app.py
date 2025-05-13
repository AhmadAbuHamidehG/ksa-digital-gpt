
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

# إعداد المفتاح من إعدادات Streamlit Cloud
openai_key = st.secrets["openai_api_key"]
os.environ["OPENAI_API_KEY"] = openai_key

# تحميل قاعدة المعرفة من Chroma
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory="ksa_knowledge",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever()

# بناء نظام السؤال والجواب
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    retriever=retriever
)

# واجهة Streamlit
st.set_page_config(page_title="المستشار الرقمي", layout="centered")
st.title("📘 المستشار الرقمي - لوائح وأنظمة")
st.markdown("أدخل سؤالك بناءً على السياسات والأنظمة الرقمية السعودية التي ستتم إضافتها لاحقًا.")

query = st.text_input("✍️ ما هو سؤالك؟")

if query:
    with st.spinner("يتم المعالجة..."):
        result = qa.run(query)
        st.success("✅ الإجابة:")
        st.write(result)
