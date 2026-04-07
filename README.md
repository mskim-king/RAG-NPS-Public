# RAG-NPS-Public

Streamlit UI 기초 예제입니다.  
이 버전은 RAG/LLM 로직 없이 채팅 UI 동작만 포함합니다.

## 1. 설치

```bash
pip install -r requirements.txt
```

## 2. 실행

```bash
streamlit run app.py
```

## 3. 사용 방법

1. 좌측 사이드바에서 PDF 파일을 업로드합니다.
2. 우측 채팅창에 질문을 입력합니다.
3. 질문마다 어시스턴트 답변 `"안녕하세요."`가 누적됩니다.
4. `대화 초기화` 버튼으로 채팅 내역을 비울 수 있습니다.
5. 모델을 선택하여 사용 할 수 있습니다. 