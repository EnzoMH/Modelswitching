import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
import asyncio
from typing import List
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from huggingface_hub import InferenceClient
from llama_cpp import Llama
from transformers import AutoTokenizer  

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 정적 파일 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Streaming 콜백 핸들러
class StreamingCallback(StreamingStdOutCallbackHandler):
    def __init__(self, websocket):
        super().__init__()
        self.websocket = websocket
        self.buffer = ""
        
    def on_llm_start(self, *args, **kwargs):
        print("AI가 대화를 시작합니다.")
        
    def on_llm_end(self, *args, **kwargs):
        print("AI가 대화를 종료합니다.")

    async def on_llm_new_token(self, token: str, **kwargs):
        print(f"New token: {token}")
        self.buffer += token
        # 토큰을 즉시 전송하도록 수정
        chunk_message = {
            "type": "assistant",
            "content": token,
            "streaming": True
        }
        try:
            await self.websocket.send_text(json.dumps(chunk_message))
        except Exception as e:
            print(f"Error sending message: {str(e)}")

# # Ollama 모델 초기화
# try:
#     llm = ChatOllama(
#         model="EEVE-Korean-10.8B:latest",
#         callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
#         temperature=0.8,
#         max_tokens=1000
#     )
#     logger.info("Ollama 모델 초기화 성공")
# except Exception as e:
#     logger.error(f"Ollama 모델 초기화 실패: {str(e)}")
#     raise

def create_assistant_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", """당신은 아래의 특성을 가진 도움이 되는 AI 조수입니다:
            1. 사용자의 질문에 직접적으로 답변하기
            2. 필요한 경우에만 부가 설명 제공하기
            3. 모호한 경우 구체적인 질문하기"""),
        ("system", "대화 맥락:\n{chat_history}"),
        ("human", "{input}")
    ])

# 프롬프트 템플릿 생성
assistant_prompt = create_assistant_prompt()

# WebSocket 연결 관리
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.conversation_histories: dict = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.conversation_histories[websocket] = []

    def disconnect(self, websocket: WebSocket):
        if websocket in self.conversation_histories:
            del self.conversation_histories[websocket]
        self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    def get_conversation_history(self, websocket: WebSocket) -> List[dict]:
        return self.conversation_histories.get(websocket, [])

    def add_to_history(self, websocket: WebSocket, message: dict):
        if websocket not in self.conversation_histories:
            self.conversation_histories[websocket] = []
        self.conversation_histories[websocket].append(message)

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(static_dir, "home.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# API 클라이언트 초기화

# 모델 경로 설정
MODEL_PATH = r""

# 토크나이저 초기화를 전역으로 이동하고 캐싱
tokenizer = None

def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained('Bllossom/llama-3.2-Korean-Bllossom-3B', use_fast=True)
    return tokenizer

# 모델과 토크나이저 초기화
try:
    model = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,  # 컨텍스트 크기
    n_threads=12,  # CPU 스레드 수 증가
    n_batch=1024,  # 배치 크기 증가
    n_gpu_layers=33  # GPU 레이어 활용 (GPU가 있는 경우)
    )
    
    tokenizer = AutoTokenizer.from_pretrained('Bllossom/llama-3.2-Korean-Bllossom-3B')
    logger.info("모델과 토크나이저 초기화 성공")
except Exception as e:
    logger.error(f"모델 초기화 실패: {str(e)}")
    raise

async def generate_ai_response(prompt: str, history: List[dict], websocket: WebSocket) -> str:
    try:
        # 이전 대화 컨텍스트 제한
        recent_history = history[-3:]  # 최근 3개 메시지만 사용
        
        messages = []
        for msg in recent_history:
            role = "assistant" if msg["type"] == "assistant" else "user"
            messages.append({"role": role, "content": msg["content"]})
        
        messages.append({"role": "user", "content": prompt})
        
        # 토크나이저 최적화
        tokenizer = get_tokenizer()
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 생성 파라미터 최적화
        generation_kwargs = {
            "max_tokens": 512,
            "stop": ["<|eot_id|>"],
            "echo": False,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": True,
            "repeat_penalty": 1.1,  # 반복 페널티 추가
            "tfs_z": 0.95,  # 테일 프리 샘플링
            "top_k": 40,    # Top-K 샘플링
            "mirostat_mode": 2,  # Mirostat 샘플링
            "mirostat_tau": 5.0,
            "mirostat_eta": 0.1
        }
        
        response_stream = model(formatted_prompt, **generation_kwargs)
        full_response = ""
        
        # 청크 크기 증가
        chunk_buffer = ""
        for response in response_stream:
            if 'choices' in response and len(response['choices']) > 0:
                chunk = response['choices'][0]['text']
                if chunk:
                    chunk_buffer += chunk
                    # 버퍼가 일정 크기에 도달하면 전송
                    if len(chunk_buffer) >= 10:  # 청크 크기 조정
                        chunk_message = {
                            "type": "assistant",
                            "content": chunk_buffer,
                            "streaming": True
                        }
                        await websocket.send_text(json.dumps(chunk_message))
                        full_response += chunk_buffer
                        chunk_buffer = ""
        
        # 남은 버퍼 전송
        if chunk_buffer:
            chunk_message = {
                "type": "assistant",
                "content": chunk_buffer,
                "streaming": True
            }
            await websocket.send_text(json.dumps(chunk_message))
            full_response += chunk_buffer
        
        return full_response
        
    except Exception as e:
        logger.error(f"AI 응답 생성 중 오류 발생: {str(e)}")
        raise
    
    
# WebSocket 엔드포인트 수정
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("content", "")

            user_message_dict = {"type": "user", "content": user_message}
            manager.add_to_history(websocket, user_message_dict)

            try:
                history = manager.get_conversation_history(websocket)
                
                # AI 응답 생성
                ai_response = await generate_ai_response(
                    user_message,
                    history,
                    websocket
                )
                
                # 완료 메시지 전송
                complete_message = {
                    "type": "assistant",
                    "content": "",
                    "streaming": False
                }
                await manager.send_message(json.dumps(complete_message), websocket)

                # 전체 응답을 히스토리에 추가
                ai_message = {
                    "type": "assistant",
                    "content": ai_response
                }
                manager.add_to_history(websocket, ai_message)

            except Exception as e:
                error_message = {
                    "type": "error",
                    "content": f"오류가 발생했습니다: {str(e)}"
                }
                await manager.send_message(json.dumps(error_message), websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
