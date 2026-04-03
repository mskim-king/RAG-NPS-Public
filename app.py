# app.py
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from vector_store import build_vector_store
from langchain.tools import tool
from vector_store import get_retriever
from langchain.agents import create_agent

@tool
def rag_tool(query: str):
    """
    2025년 국민연금기금의 운용수익률 및 자산 포트폴리오 성과 데이터를 검색하는 도구입니다.
    """
    retriever = get_retriever()
    docs = retriever.invoke(query)
    print(docs)

    return "\n\n".join([doc.page_content for doc in docs])

load_dotenv()

@st.cache_resource(show_spinner=False)
def get_agent():
    return create_agent(
        model="gpt-5.4-mini",
        tools = [rag_tool],
        system_prompt = """당신은 국민연금 기금 관련 질문에 답변하는 전문 어시스턴트입니다. """
    )
# agent = create_agent(model="gpt-5.4-mini", tools=tools)

# llm = init_chat_model('gpt-5.4-nano')

# 업로드 경로 생성 함수
def save_uploaded_file(uploaded_file):
    upload_dir = Path("./uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    return str(file_path)

# 사이드바
def render_sidebar():
    with st.sidebar:
        api_key_input = st.text_input("OpenAI API Key", type="password")
        if api_key_input:
            st.session_state.openai_api_key = api_key_input
            os.environ["OPENAI_API_KEY"] = api_key_input

        uploaded_files = st.file_uploader(
            "파일 업로드",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("벡터스토어 생성"):
            file_path = save_uploaded_file(uploaded_files[0])
            result = build_vector_store(file_path)
            st.session_state.vector_store_ready = True
            st.success(result)
            st.session_state.uploaded_files_meta = [
                {"name": file.name, "size": file.size} for file in uploaded_files
            ]
        else:
            st.session_state.uploaded_files_meta = []

        st.subheader("업로드된 파일")
        if st.session_state.uploaded_files_meta:
            for item in st.session_state.uploaded_files_meta:
                size_kb = item["size"] / 1024
                st.write(f"- {item['name']} ({size_kb:.1f} KB)")
        else:
            st.caption("아직 업로드된 파일이 없습니다.")

        if st.button("대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# 채팅 내역
def render_chat():
    st.title("NPS X RAG")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    query = st.chat_input("질문을 입력해 주세요.")
    if not query:
        return
    # response = llm.invoke(query)
    # answer = response.content
    agent = get_agent()
    response = agent.invoke(
        {"messages"	:	[{"role":	"user",	"content":	query}]}
    )

    answer	=	response['messages'][-1].content

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()


st.set_page_config(page_title="기초 챗봇 UI", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files_meta" not in st.session_state:
    st.session_state.uploaded_files_meta = []

render_sidebar()
render_chat()
