# config.py
import os
from huggingface_hub import hf_hub_download

# 네이버 API 설정
# 직접 발급 받은 네이버 API 키로 설정하여주세요.
NAVER_CLIENT_ID = 'gJoWhDFkqBz1Mn57jQDx'
NAVER_CLIENT_SECRET = '9q7TO0IxMB'

# 모델 설정
YOLO_MODEL_PATH = hf_hub_download("Bingsu/adetailer", "deepfashion2_yolov8s-seg.pt")  # DeepFashion2 사전학습 모델
LLM_MODEL_PATH = "Bllossom/llama-3.2-Korean-Bllossom-3B"
EMBEDDING_MODEL_PATH = "jhgan/ko-sroberta-multitask"
CLIP_MODEL_PATH = "openai/clip-vit-base-patch32"

# ChromaDB 설정
CHROMA_PERSIST_DIR = "./chroma_db"

# 크롤링 설정
CRAWL_CATEGORIES = ['셔츠', '바지', '원피스', '자켓', '스커트', '니트', '코트']
CRAWL_MAX_PAGES = 3

# 검색 설정
MAX_SEARCH_COUNT = 10  # 최대 API 검색 횟수
SEARCH_DISPLAY_COUNT = 10  # 한 번에 가져올 상품 수
VECTOR_SEARCH_K = 5  # 벡터 스토어에서 검색할 문서 수

# 토큰 관리 설정
MAX_CONTEXT_TOKENS = 2048  # 최대 컨텍스트 토큰 수
MAX_GENERATION_TOKENS = 512  # 최대 생성 토큰 수

# Gradio 서버 설정
GRADIO_SERVER_PORT = 7860
GRADIO_SERVER_NAME = "0.0.0.0"  # 모든 IP에서 접근 가능, "127.0.0.1"로 변경하면 로컬만 접근 가능