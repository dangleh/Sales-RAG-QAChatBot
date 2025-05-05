__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
import csv

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# --- INITIAL SETUP ---
# Load embeddings and vectorstore
@st.cache_resource
def load_vectorstore():
    embedding = OpenAIEmbeddings()
    docs = []
    with open("service_quote_media.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Gom thông tin từng dịch vụ thành tập text
            text = (f"STT: {row['STT']}\n"
                    f"Dịch vụ: {row['Tên Dịch Vụ']}\n"
                    f"Mô tả: {row['Mô Tả Ngắn']}\n"
                    f"Số lượng: {row['Số Lượng']}\n"
                    f"Đơn giá: {row['Đơn Giá (₫)']}\n")
            docs.append(text)
    vector_store = Chroma.from_texts(texts=docs, embedding=embedding, persist_directory="./chroma_media")
    return vector_store.as_retriever(search_kwargs={"k": 3})

template = """
Bạn là Piccolo Media Bot – trợ lý bán hàng chuyên về tư vấn dịch vụ media production cho Piccolo Media.
Hãy trả lời dựa trên thông tin báo giá được truy xuất dưới đây, đảm bảo:
- Phong cách trả lời tự nhiên, giống như một người thật. Xưng hô "Anh/Chị" với người hỏi
- Dùng tiếng Việt.
- Nếu không có đủ dữ liệu, đáp: "Xin lỗi, em chưa rõ lắm." hoặc "Chờ em hỏi lại anh Quynh nhé!".
- Không đoán hoặc bịa ra câu trả lời.

Thông tin báo giá (context):
{context}

Khách hỏi: {question}
Bot trả lời:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Build the RAG QA chain@st.cache_resource
def build_qa_chain(retriever):
    llm = OpenAI(temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

st.title("Piccolo Sales Bot")
st.write("Anh/chị vui lòng cho biết nhu cầu của bạn để được tư vấn.")

retriever = load_vectorstore()
qa_chain = build_qa_chain(retriever)

def main():
    question = st.text_input("Khách hỏi:")
    if question:
        with st.spinner("Anh/chị chờ em tí ạ..."):
            answer = qa_chain.invoke(question)
        st.markdown(f"{answer['result']}")

if __name__ == "__main__":
    main()