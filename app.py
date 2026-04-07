# 앱 실행 스크립트
import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent

from vector_store import build_vector_store, get_retriever

# OpenAI API Key 기본 형식 검사 정규식
OPENAI_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]{20,}$")

# OpenAI 모델 목록 및 특징
AVAILABLE_MODELS = {
    "gpt-4o": {
        "name": "GPT-4o (권장)",
        "description": "최신, 가장 강력한 성능. 복잡한 분석과 정확도가 중요한 작업에 최적"
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini (빠르고 저렴)",
        "description": "빠른 응답과 비용 효율적. 대부분의 일반적인 작업에 추천"
    },
    "gpt-4-turbo": {
        "name": "GPT-4 Turbo (고성능)",
        "description": "매우 강력한 성능. 복잡한 추론과 정확한 응답이 필요한 경우"
    },
    "gpt-3.5-turbo": {
        "name": "GPT-3.5 Turbo (가장 저렴)",
        "description": "가장 빠르고 저렴. 간단한 작업과 캐시 활용에 좋음"
    }
}


# OpenAI API Key 유효성 확인 함수
# 입력된 키가 올바른 형식을 갖추었는지 판별합니다.
def is_valid_openai_key(key: str) -> bool:
    if not isinstance(key, str):
        return False
    return bool(OPENAI_KEY_PATTERN.fullmatch(key.strip()))

# RAG 검색 도구 정의
# LangChain 도구로 사용되어 질의에 대한 관련 문서를 검색합니다.
@tool
def rag_tool(query: str):
    """
    2025년 국민연금기금의 운용수익률 및 자산 포트폴리오 성과 데이터를 검색하는 도구입니다.
    """
    retriever = get_retriever()
    docs = retriever.get_relevant_documents(query)

    if not docs:
        return "관련 문서를 찾을 수 없습니다. 다른 질문으로 시도해 주세요."

    return "\n\n".join([doc.page_content for doc in docs])

# .env 파일에서 환경 변수를 로드합니다.
# 로컬 환경에서는 .env에 있는 OPENAI_API_KEY를 자동으로 사용합니다.
# 서버 환경에서는 수동 입력을 요구하도록 APP_ENV를 통해 제어합니다.
load_dotenv()
APP_ENV = os.getenv("APP_ENV", "local").strip().lower()
USE_LOCAL_ENV_KEY = APP_ENV != "server"

# LangChain 에이전트를 생성하고 API 키별로 캐시합니다.
# OpenAI API Key가 변경되면 새로운 에이전트가 생성됩니다.
@st.cache_resource(show_spinner=False)
def get_agent(api_key: str, model: str = "gpt-4o-mini"):
    if not api_key:
        raise ValueError("OpenAI API Key가 필요합니다.")

    os.environ["OPENAI_API_KEY"] = api_key
    return create_agent(
        model=model,
        tools=[rag_tool],
        system_prompt="당신은 국민연금 기금 관련 질문에 답변하는 전문 어시스턴트입니다.",
    )

# 업로드 파일을 저장할 디렉터리를 생성하고 파일을 저장하는 함수
def save_uploaded_file(uploaded_file):
    upload_dir = Path("./uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    return str(file_path)


# Streamlit 세션 상태 초기화 함수
# 메시지, 업로드 정보, API 키 상태를 기본값으로 설정합니다.
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_files_meta" not in st.session_state:
        st.session_state.uploaded_files_meta = []
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    if "api_key_error" not in st.session_state:
        st.session_state.api_key_error = ""
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gpt-4o-mini"

    # 로컬 환경에서 .env에 OPENAI_API_KEY가 설정되어 있으면 자동으로 불러옵니다.
    if USE_LOCAL_ENV_KEY and not st.session_state.openai_api_key:
        env_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if env_api_key:
            if is_valid_openai_key(env_api_key):
                st.session_state.openai_api_key = env_api_key
                os.environ["OPENAI_API_KEY"] = env_api_key
            else:
                st.session_state.api_key_error = "로컬 .env에 설정된 OpenAI API Key가 유효하지 않습니다. 확인 후 수정해 주세요."

# 사이드바 UI 렌더링 함수
def render_sidebar():
    with st.sidebar:
        st.header("설정")

        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key,
        )
        api_key_input = api_key_input.strip() if api_key_input else api_key_input

        if api_key_input != st.session_state.openai_api_key:
            if api_key_input and is_valid_openai_key(api_key_input):
                st.session_state.openai_api_key = api_key_input
                os.environ["OPENAI_API_KEY"] = api_key_input
                st.session_state.api_key_error = ""
            elif api_key_input:
                st.session_state.api_key_error = "OpenAI API Key가 유효하지 않습니다. 다시 입력해 주세요."

        if st.session_state.api_key_error:
            st.error(st.session_state.api_key_error)
        elif st.session_state.openai_api_key:
            st.success(
                "OpenAI API Key가 형식상 유효합니다. 실제 인증은 질문 전송 시 다시 확인됩니다."
            )

        st.divider()
        st.subheader("모델 선택")
        
        # 모델 선택 드롭다운
        model_options = list(AVAILABLE_MODELS.keys())
        selected_model_key = st.selectbox(
            "사용할 모델을 선택하세요",
            options=model_options,
            format_func=lambda x: AVAILABLE_MODELS[x]["name"],
            index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 1
        )
        
        if selected_model_key != st.session_state.selected_model:
            st.session_state.selected_model = selected_model_key
            # 모델 변경 시 캐시 초기화
            st.cache_resource.clear()
        
        # 선택된 모델 설명 표시
        st.info(f"📌 {AVAILABLE_MODELS[selected_model_key]['description']}")

        if APP_ENV == "server":
            st.info("서버 모드에서는 OPENAI API Key를 수동 입력해야 합니다.")
        else:
            st.info("로컬 모드에서는 .env에 설정된 OPENAI API Key를 자동으로 사용합니다.")

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
        elif not uploaded_files:
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

# 채팅 입력과 응답 표시를 담당하는 화면 구성 함수
# 사용자 질문을 받아 LangChain 에이전트를 실행하고 결과를 보여줍니다.
# OpenAI API Key 검사가 먼저 수행됩니다.
def render_chat():
    st.title("NPS X RAG")

    if not st.session_state.openai_api_key or not is_valid_openai_key(st.session_state.openai_api_key):
        st.warning("유효한 OpenAI API Key를 입력해야 질문에 답변할 수 있습니다.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    query = st.chat_input("질문을 입력해 주세요.")
    if not query:
        return

    if not st.session_state.openai_api_key or not is_valid_openai_key(st.session_state.openai_api_key):
        st.warning("OpenAI API Key가 유효하지 않습니다. 설정에서 다시 입력해 주세요.")
        return

    # 아래는 LangChain 에이전트 호출 예시입니다.
    # 실제 실행 시 OpenAI API 키가 올바르게 설정되어 있어야 합니다.
    agent = get_agent(st.session_state.openai_api_key, st.session_state.selected_model)
    try:
        response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    except Exception as e:
        error_message = str(e)
        if "invalid_api_key" in error_message or "Incorrect API key" in error_message:
            st.error(
                "OpenAI API Key 인증에 실패했습니다. 입력한 키가 정확한지 확인하고, "
                "https://platform.openai.com/account/api-keys 에서 새 키를 발급받아 다시 입력해 주세요."
            )
        else:
            st.error(f"질문 처리 중 오류가 발생했습니다: {e}")
        return

    def extract_message_text(message):
        if hasattr(message, "content"):
            return message.content
        if isinstance(message, dict):
            return message.get("content", str(message))
        return str(message)

    if isinstance(response, dict) and "messages" in response:
        last_message = response["messages"][-1]
        answer = extract_message_text(last_message)
    elif isinstance(response, list) and response:
        last_message = response[-1]
        answer = extract_message_text(last_message)
    elif hasattr(response, "content"):
        answer = response.content
    else:
        answer = str(response)

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()


# Streamlit 페이지 기본 구성
st.set_page_config(page_title="기초 챗봇 UI", layout="wide")

# 세션 상태 초기화 후 UI 렌더링
initialize_session_state()
render_sidebar()
render_chat()
