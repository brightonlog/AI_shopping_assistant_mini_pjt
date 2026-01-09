# app.py
import gradio as gr
import torch
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
# CLIP 관련 imports를 transformers로 대체
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
import requests
import os
from config import *
from utils import TokenManager

# 실습과제2 : regex 라이브러리 import 하기
import re

# 전역 변수
models = {}
vector_store = None
image_vectorstore = None  # CLIP 이미지 벡터 스토어
image_vectors_cache = []  # CLIP 벡터 캐시 (numpy 배열)
image_metadata_cache = []  # 상품 메타데이터 캐시
conversation_chain = None
search_count = 0
conversation_history = []
token_manager = None

def load_models():
    """
    모든 AI 모델을 로드하고 초기화합니다.
    - YOLO: 패션 아이템 객체 탐지
    - CLIP: 이미지-텍스트 임베딩
    - LLM: 대화형 AI
    - Vector Store: 상품 정보 저장 및 검색
    """
    global models, vector_store, conversation_chain, token_manager
    
    # YOLOv8 DeepFashion2 모델 로드
    print("YOLO 모델 로딩 중...")
    models['yolo'] = YOLO(YOLO_MODEL_PATH, task='segment')
    
    # CLIP 모델 로드 (transformers 라이브러리 사용)
    print("CLIP 모델 로딩 중...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models['device'] = device
    
    # Hugging Face의 CLIP 모델 사용
    # 기본적으로 openai/clip-vit-base-patch32 모델 사용
    clip_model_name = CLIP_MODEL_PATH
    models['clip_processor'] = CLIPProcessor.from_pretrained(clip_model_name, use_fast=True)
    models['clip_model'] = CLIPModel.from_pretrained(clip_model_name).to(device)
    
    # LLM 모델 로드
    print("LLM 모델 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
    # pad_token이 없는 경우 eos_token으로 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # GPU 사용 명시적 설정
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    
    # GPU 사용 확인
    if torch.cuda.is_available():
        print(f"LLM이 GPU에서 실행됩니다: {torch.cuda.get_device_name()}")
    else:
        print("LLM이 CPU에서 실행됩니다")
    
    # TokenManager 초기화
    token_manager = TokenManager(tokenizer)
    
    # LangChain 설정
    pipeline_kwargs = {
        "temperature": 0.7, 
        "max_length": 512,
        "do_sample": True,
        "top_p": 0.9
    }
    
    llm = HuggingFacePipeline.from_model_id(
        model_id=LLM_MODEL_PATH,
        task="text-generation",
        model_kwargs=pipeline_kwargs
    )
    
    # 벡터 스토어 초기화
    print("벡터 스토어 초기화 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH
    )
    
    # ChromaDB 디렉토리가 없으면 생성
    if not os.path.exists(CHROMA_PERSIST_DIR):
        os.makedirs(CHROMA_PERSIST_DIR)
    
    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )
    
    # 대화 체인 설정
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": VECTOR_SEARCH_K}),
        memory=memory,
        return_source_documents=True
    )
    
    models['tokenizer'] = tokenizer
    models['llm'] = model
    
    # 이미지 벡터 스토어 초기화
    print("이미지 벡터 스토어 초기화 중...")
    setup_image_vectorstore()
    
    print("모든 모델 로딩 완료!")
    return models

def setup_image_vectorstore():
    """
    CLIP 이미지 벡터 저장을 위한 ChromaDB 별도 Collection 생성
    """
    global image_vectorstore
    
    # 이미지 벡터 저장 디렉토리
    image_persist_dir = "./chroma_db_images"
    
    if not os.path.exists(image_persist_dir):
        os.makedirs(image_persist_dir)
    
    # HuggingFace 임베딩 (텍스트용은 기존 것 사용, 이미지는 직접 CLIP 벡터 사용)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH
    )
    
    image_vectorstore = Chroma(
        collection_name="product_images",
        persist_directory=image_persist_dir,
        embedding_function=embeddings
    )
    
    print("이미지 벡터 스토어 초기화 완료!")

def download_and_preprocess_image(image_url):
    """
    상품 이미지 URL에서 이미지 다운로드 및 전처리
    """
    try:
        response = requests.get(image_url, timeout=5)
        if response.status_code == 200:
            from io import BytesIO
            image = Image.open(BytesIO(response.content))
            # RGB로 변환 (RGBA 등 다른 모드 대응)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
    except Exception as e:
        print(f"이미지 다운로드 실패 ({image_url}): {e}")
    return None

def has_valid_image_url(product):
    """
    상품의 유효한 이미지 URL 인지 체크 (placeholder 이미지 제외)
    """
    image_url = product.get('image', '')
    return image_url and 'placeholder' not in image_url and image_url.strip() != ''

def detect_clothing_color(image, bbox):
    """
    탐지된 옷 영역에서 주요 색상을 k-means++ clustering으로 추출합니다.
    
    Args:
        image: PIL Image 객체
        bbox: [x1, y1, x2, y2] 바운딩 박스 좌표
        
    Returns:
        color_name: 주요 색상 이름
        color_rgb: RGB 값 튜플
    """
    try:
        # 이미지를 numpy 배열로 변환
        img_array = np.array(image)
        
        # 바운딩 박스 영역 추출
        x1, y1, x2, y2 = map(int, bbox)
        cropped = img_array[y1:y2, x1:x2]
        
        # 픽셀 데이터를 2D 배열로 변환 (각 행이 하나의 픽셀 RGB 값)
        pixels = cropped.reshape(-1, 3).astype(np.float32)
        
        # k-means++ 클러스터링 (OpenCV 사용)
        k = 5  # 클러스터 개수
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        
        # k-means++ 초기화 방식 사용
        _, labels, centers = cv2.kmeans(
            pixels, 
            k, 
            None, 
            criteria, 
            10, 
            cv2.KMEANS_PP_CENTERS  # k-means++ 초기화
        )
        
        # 가장 많이 나타난 클러스터의 중심 색상 찾기
        unique, counts = np.unique(labels, return_counts=True)
        dominant_cluster = unique[np.argmax(counts)]
        dominant_color = centers[dominant_cluster].astype(int)
        
        # RGB 값을 색상 이름으로 변환
        color_name = rgb_to_color_name(dominant_color)
        
        return color_name, tuple(dominant_color)
        
    except Exception as e:
        print(f"색상 추출 오류: {e}")
        return "알 수 없음", (128, 128, 128)

def rgb_to_color_name(rgb):
    """
    RGB 값을 가장 가까운 색상 이름으로 변환합니다.
    HSV 색공간을 사용하여 더 정확한 색상 판별을 수행합니다.
    
    Args:
        rgb: RGB 값 배열 [R, G, B]
        
    Returns:
        color_name: 색상 이름
    """
    # RGB를 HSV로 변환 (OpenCV는 BGR 사용하므로 변환 필요)
    rgb_normalized = np.uint8([[rgb]])
    hsv = cv2.cvtColor(rgb_normalized, cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = hsv[0], hsv[1], hsv[2]
    
    # 무채색 판별 (채도가 낮은 경우)
    if s < 30:  # 채도가 매우 낮음
        if v < 50:
            return '검정'
        elif v > 200:
            return '흰색'
        elif v < 100:
            return '진회색'
        else:
            return '회색'
    
    # 베이지/카키 판별 (낮은 채도 + 특정 색상 범위)
    if s < 60:
        if 15 <= h <= 35 and v > 180:
            return '베이지'
        elif 20 <= h <= 45 and 120 < v <= 200:
            return '카키'
    
    # HSV 기반 색상 정의 (Hue 범위, 채도 범위, 명도 범위)
    # OpenCV에서 H는 0-180 범위
    color_ranges = {
        '빨강': [(0, 10, 70, 255, 100, 255), (170, 180, 70, 255, 100, 255)],  # 빨강은 0과 180 근처
        '진빨강': [(0, 10, 80, 255, 50, 120), (170, 180, 80, 255, 50, 120)],
        '주황': [(10, 20, 70, 255, 100, 255)],
        '노랑': [(20, 35, 70, 255, 150, 255)],
        '연두': [(35, 50, 60, 255, 100, 255)],
        '초록': [(50, 85, 60, 255, 50, 200)],
        '청록': [(85, 100, 60, 255, 100, 255)],
        '하늘': [(100, 120, 40, 150, 150, 255)],
        '파랑': [(100, 130, 70, 255, 80, 200)],
        '네이비': [(100, 130, 60, 255, 30, 80)],
        '남색': [(130, 145, 60, 255, 50, 150)],
        '보라': [(130, 155, 60, 255, 80, 200)],
        '연보라': [(130, 160, 30, 80, 180, 255)],
        '자주': [(145, 165, 70, 255, 50, 150)],
        '분홍': [(160, 180, 30, 100, 200, 255)],
        '진분홍': [(160, 175, 70, 255, 150, 255)],
        '갈색': [(10, 25, 60, 255, 30, 100)],
        '와인': [(160, 180, 60, 255, 30, 100)]
    }
    
    # 각 색상 범위와 비교하여 매칭
    for color_name, ranges in color_ranges.items():
        for color_range in ranges:
            h_min, h_max, s_min, s_max, v_min, v_max = color_range
            if h_min <= h <= h_max and s_min <= s <= s_max and v_min <= v <= v_max:
                return color_name
    
    # 매칭되지 않으면 Hue 값으로 대략적인 색상 반환
    if h < 10 or h >= 170:
        return '빨강'
    elif h < 20:
        return '주황'
    elif h < 35:
        return '노랑'
    elif h < 85:
        return '초록'
    elif h < 130:
        return '파랑'
    else:
        return '보라'

def detect_fashion_objects(image):
    """
    업로드된 이미지에서 패션 아이템을 탐지합니다. (CLIP 통합 버전)
    
    Args:
        image: PIL Image 객체
        
    Returns:
        img_with_boxes: 바운딩 박스가 그려진 이미지
        html_output: HTML 형식의 탐지 결과
    """
    global conversation_history
    
    if image is None:
        return None, "<p>이미지를 업로드해주세요.</p>"
    
    try:
        # YOLO 모델로 객체 탐지 실행
        results = models['yolo'](image)
        
        # 이미지 복사본에 바운딩 박스 그리기
        img_with_boxes = np.array(image).copy()
        detected_items = []
        
        # 디버깅 출력
        print(f"YOLO 탐지 결과 수: {len(results)}")
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                print(f"박스 수: {len(boxes)}")
                for box in boxes:
                    # 바운딩 박스 좌표 추출
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()  # 신뢰도
                    cls = int(box.cls[0].item())  # 클래스 ID
                    class_name = models['yolo'].names[cls]  # 클래스 이름
                    
                    # k-means++ 클러스터링으로 색상 추출
                    color_name, color_rgb = detect_clothing_color(image, [x1, y1, x2, y2])
                    
                    print(f"탐지된 아이템: {class_name} (신뢰도: {conf:.2f}, 색상: {color_name})")
                    
                    # 바운딩 박스와 레이블 그리기
                    cv2.rectangle(img_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(img_with_boxes, f"{class_name} ({color_name}) {conf:.2f}", 
                               (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    detected_items.append(f"{color_name} {class_name}")
        
        # PIL Image로 변환
        img_with_boxes = Image.fromarray(img_with_boxes)
        
        if detected_items:
            print(f"총 탐지된 아이템: {detected_items}")
            
            # 1. 텍스트 기반 검색 (기존)
            text_products = search_products(' '.join(detected_items[:2]))
            save_products_to_vectorstore(text_products)
            
            # 2. CLIP 이미지 유사도 검색 (신규)
            print("CLIP 이미지 유사도 검색 중...")
            similar_products = search_similar_images(image, k=10)  # 10개로 증가
            
            # 3. 검색된 상품들을 CLIP 벡터로 저장
            save_products_with_clip_vectors(text_products)
            
            # 4. 결과 병합 및 중복 제거 (CLIP 검색 결과를 앞에 배치)
            combined_products = merge_and_deduplicate_products(similar_products, text_products)
            print(f"병합된 상품 수: {len(combined_products)} (CLIP: {len(similar_products)}, 텍스트: {len(text_products)})")
            
            # HTML 출력 생성
            html_output = f"<h3>탐지된 아이템</h3><p>{', '.join(detected_items)}</p>"
            html_output += f"<p style='color: #666; font-size: 12px;'>텍스트 검색: {len(text_products)}개 | CLIP 검색: {len(similar_products)}개 | 총: {len(combined_products)}개</p><hr>"
            html_output += format_product_html(combined_products)
            
            # LLM 초기 메시지 생성
            detected_items_str = ', '.join(detected_items)
            
            try:
                # 간단한 초기 인사 메시지 생성
                llm_response = f"안녕하세요! 이미지에서 {detected_items_str}을(를) 발견했습니다. 텍스트 검색과 이미지 유사도 검색을 통해 {len(combined_products)}개의 상품을 찾았어요. 어떤 스타일이나 브랜드를 선호하시나요? 예산도 알려주시면 더 정확한 추천을 도와드릴 수 있어요."
                
                # 대화 히스토리에 추가
                conversation_history.append({"role": "assistant", "content": llm_response})
            except Exception as e:
                print(f"LLM 응답 생성 오류: {e}")
                # LLM 오류 시 기본 메시지 사용
                default_message = f"안녕하세요! 이미지에서 {detected_items_str}을(를) 발견했습니다. 어떤 스타일의 상품을 찾고 계신가요?"
                conversation_history.append({"role": "assistant", "content": default_message})
        else:
            html_output = "<p>패션 아이템을 찾을 수 없습니다.</p>"
        
        return img_with_boxes, html_output
        
    except Exception as e:
        print(f"이미지 탐지 오류: {e}")
        import traceback
        traceback.print_exc()
        return image, f"<p>오류가 발생했습니다: {str(e)}</p>"

def extract_clip_features(image):
    """
    CLIP 모델을 사용하여 이미지의 특징 벡터를 추출합니다.
    
    Args:
        image: PIL Image 객체
        
    Returns:
        features: 이미지 특징 벡터 (numpy array)
    """
    # CLIP 프로세서로 이미지 전처리
    inputs = models['clip_processor'](images=image, return_tensors="pt")
    inputs = {k: v.to(models['device']) for k, v in inputs.items()}
    
    # 특징 추출
    with torch.no_grad():
        image_features = models['clip_model'].get_image_features(**inputs)
    
    return image_features.cpu().numpy()

def save_products_with_clip_vectors(products):
    """
    네이버 API로 가져온 상품들을 CLIP 벡터로 변환하여 저장
    """
    global image_vectorstore, image_vectors_cache, image_metadata_cache
    
    if not products:
        return
    
    print(f"{len(products)}개 상품의 이미지 벡터 추출 중...")
    
    for idx, product in enumerate(products):
        if has_valid_image_url(product):
            try:
                # 이미지 다운로드
                image = download_and_preprocess_image(product['image'])
                if image is None:
                    continue
                
                # CLIP 벡터 추출
                clip_vector = extract_clip_features(image)
                clip_vector_flat = clip_vector.flatten()
                
                # 메타데이터 준비
                metadata = {
                    'title': product.get('title', ''),
                    'link': product.get('link', ''),
                    'lprice': product.get('lprice', '0'),
                    'hprice': product.get('hprice', '0'),
                    'mall': product.get('mallName', ''),
                    'image': product.get('image', ''),
                    'productId': str(product.get('productId', '')),
                    'brand': product.get('brand', ''),
                    'maker': product.get('maker', ''),
                    'category1': product.get('category1', ''),
                    'category2': product.get('category2', '')
                }
                
                # 중복 체크 - productId와 image 모두 확인
                product_id = metadata['productId']
                image_url = metadata['image']
                
                is_duplicate = False
                if product_id:
                    is_duplicate = any(m.get('productId') == product_id for m in image_metadata_cache)
                if not is_duplicate and image_url:
                    is_duplicate = any(m.get('image') == image_url for m in image_metadata_cache)
                
                if is_duplicate:
                    continue
                
                # 메모리에 저장
                image_vectors_cache.append(clip_vector_flat)
                image_metadata_cache.append(metadata)
                
            except Exception as e:
                print(f"이미지 벡터 추출 실패: {e}")
                continue
    
    print(f"{len(image_vectors_cache)}개의 이미지 벡터 저장 완료!")

def search_similar_images(user_uploaded_image, k=5):
    """
    사용자가 업로드한 이미지와 유사한 상품 이미지 검색 (CLIP 벡터 코사인 유사도)
    """
    global image_vectors_cache, image_metadata_cache
    
    if not image_vectors_cache:
        print("저장된 이미지 벡터가 없습니다.")
        return []
    
    try:
        # 사용자 이미지의 CLIP 벡터 추출
        user_clip_vector = extract_clip_features(user_uploaded_image).flatten()
        
        # 모든 저장된 벡터와 코사인 유사도 계산
        similarities = []
        for idx, stored_vector in enumerate(image_vectors_cache):
            # 코사인 유사도 = (A · B) / (||A|| * ||B||)
            dot_product = np.dot(user_clip_vector, stored_vector)
            norm_user = np.linalg.norm(user_clip_vector)
            norm_stored = np.linalg.norm(stored_vector)
            
            if norm_user > 0 and norm_stored > 0:
                similarity = dot_product / (norm_user * norm_stored)
                similarities.append((idx, similarity))
        
        # 유사도 높은 순으로 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 k개 반환
        similar_products = []
        for idx, sim_score in similarities[:k]:
            metadata = image_metadata_cache[idx]
            print(f"유사도: {sim_score:.4f} - {metadata['title'][:30]}...")
            similar_products.append({
                'title': metadata.get('title', ''),
                'link': metadata.get('link', ''),
                'lprice': metadata.get('lprice', '0'),
                'hprice': metadata.get('hprice', '0'),
                'mallName': metadata.get('mall', ''),
                'image': metadata.get('image', ''),
                'productId': metadata.get('productId', ''),
                'brand': metadata.get('brand', ''),
                'maker': metadata.get('maker', ''),
                'category1': metadata.get('category1', ''),
                'category2': metadata.get('category2', '')
            })
        
        return similar_products
        
    except Exception as e:
        print(f"이미지 검색 오류: {e}")
        import traceback
        traceback.print_exc()
        return []

def merge_and_deduplicate_products(list1, list2):
    """
    두 상품 리스트를 통합하고 중복 제거
    """
    # productId, image, title 기준으로 중복 제거
    seen_ids = set()
    seen_images = set()
    seen_titles = set()
    merged = []
    
    for product in list1 + list2:
        pid = product.get('productId', '')
        image = product.get('image', '')
        title = product.get('title', '')
        
        # 중복 체크: productId, image, title 모두 확인
        is_duplicate = False
        if pid and pid in seen_ids:
            is_duplicate = True
        if image and image in seen_images:
            is_duplicate = True
        if title and title in seen_titles:
            is_duplicate = True
        
        if not is_duplicate:
            if pid:
                seen_ids.add(pid)
            if image:
                seen_images.add(image)
            if title:
                seen_titles.add(title)
            merged.append(product)
    
    return merged

def integrate_clip_search_to_detection(image):
    """
    기존 detect_fashion_objects 함수에 CLIP 검색 통합
    """
    try:
        # 1. YOLO 탐지 (기존 로직)
        detected_items, text_products = detect_fashion_objects_only(image)
        
        # 2. CLIP 이미지 유사도 검색
        similar_products = search_similar_images(image, k=5)
        
        # 3. 새로 검색된 상품들을 CLIP 벡터로 저장
        if text_products:
            save_products_with_clip_vectors(text_products)
        
        # 4. 결과 병합
        combined_products = merge_and_deduplicate_products(text_products, similar_products)
        
        return detected_items, combined_products
        
    except Exception as e:
        print(f"CLIP 통합 검색 오류: {e}")
        return [], []

def detect_fashion_objects_only(image):
    """
    YOLO로 패션 아이템 탐지 및 텍스트 검색만 수행 (CLIP 제외)
    """
    # 기존 detect_fashion_objects 로직을 복사하되, 반환값만 수정
    try:
        results = models['yolo'](image, task='segment')
        print(f"YOLO 탐지 결과 수: {len(results)}")
        
        img_array = np.array(image)
        img_with_boxes = img_array.copy()
        
        detected_items = []
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                print(f"박스 수: {len(result.boxes)}")
                
                for i, box in enumerate(result.boxes):
                    class_id = int(box.cls.item())
                    class_name = result.names[class_id]
                    confidence = box.conf.item()
                    
                    if confidence > 0.5:
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = xyxy
                        
                        color_name, color_rgb = detect_clothing_color(image, xyxy)
                        
                        cv2.rectangle(img_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label = f"{class_name}: {confidence:.2f} ({color_name})"
                        cv2.putText(img_with_boxes, label, (int(x1), int(y1-10)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        detected_items.append(f"{color_name} {class_name}")
        
        # 탐지된 아이템으로 상품 검색
        products = []
        if detected_items:
            products = search_products(' '.join(detected_items[:2]))
            save_products_to_vectorstore(products)
        
        return detected_items, products
        
    except Exception as e:
        print(f"탐지 오류: {e}")
        return [], []


# 실습과제 2 : regex 라이브러리를 사용하여 숫자 추출하기
# 실습과제

import re

def parse_budget_request(message):
    """
    사용자의 메시지에서 예산(금액)과 필터 조건(이하, 이상, 정도 등)을 추출합니다.
    """
    # 반환할 기본 데이터 구조 
    budget_info = {
        "target_price": None,
        "condition": None
    }

    # 1. N 만원 + 키워드로 패턴 정의
    # (\d+): 숫자 그룹, ([가-힣\s]+): 뒤에 오는 한글 키워드 그룹
    budget_pattern = r"(\d+)\s*만\s*원\s*([가-힣\s]*)"
    match = re.search(budget_pattern, message)
    
    # 문자열 전체에서 패턴과 일치하는 위치 찾고, 일치하는 경우 match 객체를 반환하고 없으면 None 반환
    
    if match:
        # 만원을 10000으로 변환
        amount = int(match.group(1))
        budget_info["target_price"] = amount * 10000

        # 뒤에 붙은 키워드도 추출하기
        keyword = match.group(2).strip()

        # 키워드에 따라 가격 컨디션(이하, 정도, 이상) 설정

        if any(k in keyword for k in ["이하", "아래", "저렴한", "싼", "낮은"]):
            budget_info["condition"] = "under"
        elif "정도" in keyword or "쯤" in keyword:
            budget_info["condition"] = "around"
        elif any(k in keyword for k in ["이상", "비싼", "높은", "초과"]):
            budget_info["condition"] = "over"
        else:
            # 키워드가 딱히 없으면 기본적으로 '이하(under)'로 간주
            budget_info["condition"] = "under"

        # 숫자 없이 "더 저렴한", "더 싼"만 있는 경우 처리 (이전 기록 활용용)
    elif any(k in message for k in ["더 저렴한", "더 싼", "더 저가"]):
        budget_info["condition"] = "cheaper"

    return budget_info

# 실습과제2. 상품 리스트 필터링 및 정렬하는 함수 만들기
def apply_budget_filter(products, budget_info):
    """
    검색된 상품 리스트를 사용자의 예산 조건에 맞게 필터링하고 정렬합니다.
    """
    condition = budget_info["condition"]
    target = budget_info["target_price"]
    filtered_products = []

    # 1. '더 저렴한' 요청일 경우 기준값(target)을 이전 검색 결과의 평균가로 설정
    if condition == "cheaper" and products:
        # 모든 상품의 lprice를 숫자로 변환하여 합산
        prices = [int(p.get('lprice', 0)) for p in products]
        # 평균가 계산
        target = sum(prices) / len(prices)

    # 2. 각 상품을 돌면서 조건에 맞는지 확인 (Filtering)
    for product in products:
        # 상품 가격을 숫자로 변환 (비교를 위해 필수!)
        try:
            price = int(product.get('lprice', 0))
        except (ValueError, TypeError):
            continue
            
        # 조건별 필터링 로직 
        if condition == "under" or condition == "cheaper":
            # 설정한 금액(또는 평균가) 이하인 상품만 추가
            if price <= target:
                filtered_products.append(product)
        
        elif condition == "over":
            # 설정한 금액 이상인 상품만 추가
            if price >= target:
                filtered_products.append(product)
        
        elif condition == "around":
            # 설정한 금액의 20% 내외 범위 (0.8배 ~ 1.2배)
            if target * 0.8 <= price <= target * 1.2:
                filtered_products.append(product)
        
        else:
            # 특별한 조건이 없으면(None 등) 모든 상품을 결과에 포함
            filtered_products.append(product)

    # 3. 가격이 낮은 순으로 정렬 
    # lambda를 사용하여 lprice를 숫자로 변환한 뒤 오름차순 정렬
    final_products = sorted(filtered_products, key=lambda x: int(x.get('lprice', 0)))
    
    return final_products

def search_products(query):
    """
    네이버 쇼핑 API를 사용하여 상품을 검색합니다.
    API 호출 횟수 제한이 있으면 벡터 스토어에서 검색합니다.
    
    Args:
        query: 검색어
        
    Returns:
        products: 상품 리스트
    """
    global search_count
    
    # 검색 횟수 제한 확인
    if search_count >= MAX_SEARCH_COUNT:
        return search_from_vectorstore(query)
    
    # 네이버 API 키가 설정되지 않은 경우 더미 데이터 반환
    if not NAVER_CLIENT_ID or NAVER_CLIENT_ID == "YOUR_NAVER_CLIENT_ID":
        return generate_dummy_products(query)
    
    # 네이버 쇼핑 API 호출
    url = "https://openapi.naver.com/v1/search/shop.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    params = {"query": query, "display": SEARCH_DISPLAY_COUNT, "sort": "sim"}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            search_count += 1
            items = response.json()['items']
            
            # 데이터 정합성 처리
            # productType이 1, 2, 3인 일반상품만 필터링 (다른 값은 중고이거나, 단종 상품)
            filtered_items = []
            for item in items:
                product_type = int(item.get('productType', 0))
                if product_type in [1, 2, 3]:  # 일반상품만
                    # 데이터 정합성 처리
                    if 'image' not in item or not item['image']:
                        item['image'] = 'https://placehold.co/150'
                    
                    # 가격은 이미 숫자로 제공되므로 문자열로 변환
                    item['lprice'] = str(item.get('lprice', 0))
                    item['hprice'] = str(item.get('hprice', 0))
                    
                    # HTML 태그 제거 (title에 포함된 <b> 태그)
                    if 'title' in item:
                        item['title'] = item['title'].replace('<b>', '').replace('</b>', '')
                    
                    # 추가 필드 확인
                    item['brand'] = item.get('brand', '')
                    item['maker'] = item.get('maker', '')
                    item['category1'] = item.get('category1', '')
                    item['category2'] = item.get('category2', '')
                    
                    filtered_items.append(item)
            
            print(filtered_items)
            # CLIP 벡터도 저장
            save_products_with_clip_vectors(filtered_items)
            return filtered_items
    except Exception as e:
        print(f"API 호출 오류: {e}")
    
    return generate_dummy_products(query)

def generate_dummy_products(query):
    """
    테스트용 더미 상품 데이터를 생성합니다.
    
    Args:
        query: 검색어
        
    Returns:
        products: 더미 상품 리스트
    """
    categories = ['패션의류', '패션잡화', '화장품/미용', '디지털/가전', '가구/인테리어']
    brands = ['브랜드A', '브랜드B', '브랜드C', '브랜드D', '브랜드E']
    
    products = []
    for i in range(SEARCH_DISPLAY_COUNT):
        products.append({
            'title': f'{query} 상품 {i+1}',
            'link': f'https://example.com/product/{i+1}',
            'lprice': str(10000 + i * 5000),
            'hprice': str(15000 + i * 5000),
            'mallName': f'쇼핑몰 {i+1}',
            'image': f'https://placehold.co/150?text=Product{i+1}',
            'productId': 1000000 + i,
            'productType': 1,
            'brand': brands[i % len(brands)],
            'maker': f'제조사 {i+1}',
            'category1': categories[i % len(categories)],
            'category2': '서브카테고리'
        })
    return products

def save_products_to_vectorstore(products):
    """
    검색된 상품 정보를 벡터 스토어에 저장합니다.
    
    Args:
        products: 상품 리스트
    """
    if not products:
        return
    
    texts = []
    metadatas = []
    
    for product in products:
        # 상품 정보를 텍스트로 변환 (검색 효율성을 위해 더 많은 정보 포함)
        text_parts = [
            product.get('title', ''),
            product.get('mallName', ''),
            f"가격: {product.get('lprice', '0')}원",
            product.get('brand', ''),
            product.get('maker', ''),
            product.get('category1', ''),
            product.get('category2', '')
        ]
        text = ' '.join([part for part in text_parts if part])
        texts.append(text)
        
        # 메타데이터 저장 (모든 필드 포함)
        metadatas.append({
            'title': product.get('title', ''),
            'link': product.get('link', ''),
            'price': product.get('lprice', '0'),
            'hprice': product.get('hprice', '0'),
            'mall': product.get('mallName', ''),
            'image': product.get('image', 'https://placehold.co/150'),
            'productId': str(product.get('productId', '')),
            'brand': product.get('brand', ''),
            'maker': product.get('maker', ''),
            'category1': product.get('category1', ''),
            'category2': product.get('category2', '')
        })
    
    # 벡터 스토어에 추가
    vector_store.add_texts(texts=texts, metadatas=metadatas)

def search_from_vectorstore(query):
    """
    벡터 스토어에서 유사한 상품을 검색합니다.
    
    Args:
        query: 검색어
        
    Returns:
        products: 검색된 상품 리스트
    """
    docs = vector_store.similarity_search(query, k=SEARCH_DISPLAY_COUNT)
    products = []
    
    for doc in docs:
        if doc.metadata:
            products.append({
                'title': doc.metadata.get('title', ''),
                'link': doc.metadata.get('link', ''),
                'lprice': doc.metadata.get('lprice') or doc.metadata.get('price', '0'),  # lprice 또는 price
                'hprice': doc.metadata.get('hprice', '0'),
                'mallName': doc.metadata.get('mall', ''),
                'image': doc.metadata.get('image', 'https://placehold.co/150'),
                'productId': doc.metadata.get('productId', ''),
                'brand': doc.metadata.get('brand', ''),
                'maker': doc.metadata.get('maker', ''),
                'category1': doc.metadata.get('category1', ''),
                'category2': doc.metadata.get('category2', '')
            })
    
    return products

def format_product_list(products):
    """
    상품 리스트를 텍스트 형식으로 포맷팅합니다.
    
    Args:
        products: 상품 리스트
        
    Returns:
        formatted: 포맷팅된 상품 정보 문자열
    """
    if not products:
        return "상품을 찾을 수 없습니다."
    
    formatted = "추천 상품:\n"
    for i, product in enumerate(products[:3], 1):
        # 가격 처리 (문자열이거나 숫자일 수 있음)
        try:
            price = int(product.get('lprice', 0))
        except (ValueError, TypeError):
            price = 0
            
        formatted += f"{i}. {product.get('title', '상품명 없음')}\n"
        formatted += f"   가격: {price:,}원\n"
        formatted += f"   쇼핑몰: {product.get('mallName', '정보 없음')}\n"
        formatted += f"   구매하기: {product.get('link', '#')}\n\n"
    
    return formatted

def format_product_html(products):
    """
    상품 리스트를 HTML 형식으로 포맷팅합니다.
    
    Args:
        products: 상품 리스트
        
    Returns:
        html: HTML 형식의 상품 정보
    """
    if not products:
        return "<p>상품을 찾을 수 없습니다.</p>"
    
    html = "<h3>추천 상품</h3>"
    html += "<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; max-width: 900px;'>"
    
    for i, product in enumerate(products[:6], 1):  # 3개에서 6개로 변경
        # 가격 처리 (문자열이거나 숫자일 수 있음)
        try:
            price = int(product.get('lprice', 0))
        except (ValueError, TypeError):
            price = 0
            
        # 브랜드/제조사 정보
        brand_info = ""
        if product.get('brand'):
            brand_info = f"<p style='color: #888; font-size: 12px; margin: 2px 0;'>브랜드: {product['brand']}</p>"
        elif product.get('maker'):
            brand_info = f"<p style='color: #888; font-size: 12px; margin: 2px 0;'>제조사: {product['maker']}</p>"
        
        # 카테고리 정보
        category_info = ""
        if product.get('category1'):
            categories = [product.get('category1', '')]
            if product.get('category2'):
                categories.append(product.get('category2'))
            category_info = f"<p style='color: #999; font-size: 11px; margin: 2px 0;'>{' > '.join(categories)}</p>"
        
        html += f"""
        <div style='border: 1px solid #ddd; border-radius: 8px; padding: 12px; background: #fafafa;'>
            <img src='{product.get('image', 'https://placehold.co/150')}' 
                 style='width: 100%; height: 120px; object-fit: cover; border-radius: 5px;'
                 onerror="this.onerror=null; this.src='https://placehold.co/150';">
            <h4 style='margin: 8px 0; font-size: 13px; line-height: 1.3; height: 32px; color: black; overflow: hidden;'>{product.get('title', '')[:40]}{'...' if len(product.get('title', '')) > 40 else ''}</h4>
            {brand_info}
            {category_info}
            <p style='color: #666; margin: 5px 0; font-size: 12px;'>쇼핑몰: {product.get('mallName', '')}</p>
            <p style='font-size: 16px; font-weight: bold; color: #ff6b6b;'>
                {price:,}원
            </p>
            <a href='{product.get('link', '#')}' target='_blank' rel='noreferrer noopener' 
               style='display: inline-block; background: #007bff; color: white; 
                      padding: 6px 14px; text-decoration: none; border-radius: 4px;
                      margin-top: 8px; font-size: 13px;'>
                구매하기
            </a>
        </div>
        """
    
    html += "</div>"
    return html

def should_search_web(query):
    """
    사용자 쿼리가 웹 검색이 필요한지 판단합니다.
    
    Args:
        query: 사용자 입력
        
    Returns:
        bool: 웹 검색 필요 여부
    """
    search_keywords = ['최신', '신상', '재고', '실시간', '현재', '오늘']
    
    return any(keyword in query for keyword in search_keywords)

def extract_color_keywords(query):
    """
    검색어에서 색상 키워드 추출
    """
    color_mapping = {
        '파란': ['blue', 'navy', '네이비', '파란색', '블루'],
        '빨간': ['red', '레드', '빨강', '빨간색'],
        '하양': ['white', '화이트', '하얀', '하양색', '원피스'],
        '검정': ['black', '블랙', '검은', '검정색'],
        '노란': ['yellow', '원로우', '노랑'],
        '초록': ['green', '그린'],
        '보라': ['purple', '퍼플'],
        '분홍': ['pink', '핑크'],
        '갈색': ['brown', '브라운'],
        '회색': ['gray', 'grey', '그레이']
    }
    
    detected_colors = []
    for ko_color, variations in color_mapping.items():
        if any(v in query.lower() for v in variations):
            detected_colors.append(ko_color)
    
    return detected_colors

def generate_response_with_context(user_input, conversation_history, search_results=None):
    """
    LLM을 사용하여 대화 컨텍스트를 고려한 응답을 생성합니다.
    
    Args:
        user_input: 사용자 입력
        conversation_history: 대화 히스토리
        search_results: 검색 결과 (선택사항)
        
    Returns:
        response: 생성된 응답
    """
    tokenizer = models['tokenizer']
    model = models['llm']
    
    # 시스템 프롬프트
    system_prompt = "당신은 친절한 AI 쇼핑 어시스턴트입니다. 패션 상품에 대한 정보를 제공하고 추천합니다."
    
    # TokenManager로 프롬프트 준비
    full_prompt, prompt_tokens = token_manager.prepare_prompt(
        system_prompt=system_prompt,
        conversation_history=conversation_history,
        current_query=user_input,
        context=search_results
    )
    
    # 토큰화 및 attention_mask 생성
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    # attention_mask 명시적으로 설정
    if 'attention_mask' not in inputs:
        inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids.to(model.device),
            attention_mask=inputs.attention_mask.to(model.device),
            max_new_tokens=MAX_GENERATION_TOKENS,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 생성된 텍스트에서 프롬프트 제거 및 정리
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 입력 프롬프트 길이 계산
    input_length = len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
    
    # 생성된 부분만 추출 (입력 프롬프트 이후 텍스트)
    response = generated_text[input_length:].strip()
    
    # 응답이 너무 짧거나 비어있는 경우 기본 응답
    if len(response) < 5:
        response = "죄송합니다. 이해하지 못했습니다. 다시 한 번 말씀해 주시겠어요?"
    
    return response

# 실습과제 2: 예산 조건에 따른 챗봇 응답 생성 함수
def chat_response_with_budget(message, history):

    global search_count, conversation_history
    
    # 1. 메시지가 비어있으면 현재 히스토리 반환
    if not message or not message.strip():
        return history, ""
    
    # 2. 실습과제 2: 사용자 메시지에서 예산 조건 파싱하기
    budget_info = parse_budget_request(message)
    
    # 3. Gradio 히스토리를 내부 형식으로 변환
    if history:
        for h in history:
            if isinstance(h, list) and len(h) == 2:
                user_msg_content = h[0]
                assistant_msg_content = h[1]
                
                if not user_msg_content or not assistant_msg_content:
                    continue
                    
                user_msg_content = str(user_msg_content) if user_msg_content else ""
                assistant_msg_content = str(assistant_msg_content) if assistant_msg_content else ""
                
                user_exists = any(msg.get('content') == user_msg_content for msg in conversation_history if msg.get('role') == 'user')
                assistant_exists = any(msg.get('content') == assistant_msg_content for msg in conversation_history if msg.get('role') == 'assistant')
                
                if not user_exists and user_msg_content:
                    conversation_history.append({"role": "user", "content": user_msg_content})
                if not assistant_exists and assistant_msg_content:
                    conversation_history.append({"role": "assistant", "content": assistant_msg_content})
    
    # 4. 현재 메시지 추가 및 토큰 관리
    conversation_history, token_count = token_manager.manage_conversation_history(
        conversation_history,
        {"role": "user", "content": message}
    )
    
    # 5. 상품 검색 수행 (웹 검색 또는 벡터 DB)
    products = []
    
    # 색상 키워드 추출
    color_keywords = extract_color_keywords(message)
    
    if should_search_web(message) and search_count < MAX_SEARCH_COUNT:
        print(f'\n웹 검색 : {message}\n')
        products = search_products(message)
        save_products_to_vectorstore(products)
    else:
        print(f'\n벡터 DB 검색 : {message}\n')
        if color_keywords:
            print(f"감지된 색상: {', '.join(color_keywords)}")
        
        # 기본 검색어에 색상 키워드 추가
        search_query = message
        if color_keywords:
            # 색상 키워드를 영어로 변환하여 검색
            color_terms = []
            for color in color_keywords:
                if '파란' in color:
                    color_terms.extend(['blue', 'navy'])
                elif '하얀' in color or '하양' in color:
                    color_terms.extend(['white', 'ivory'])
                elif '검' in color:
                    color_terms.extend(['black'])
                elif '빨' in color:
                    color_terms.extend(['red'])
                elif '분홍' in color:
                    color_terms.extend(['pink'])
            search_query = f"{message} {' '.join(color_terms)}"
        
        docs = vector_store.similarity_search(search_query, k=VECTOR_SEARCH_K*2)  # 더 많이 검색
        for doc in docs:
            if doc.metadata:
                product = {
                    'title': doc.metadata.get('title', ''),
                    'link': doc.metadata.get('link', ''),
                    'lprice': doc.metadata.get('lprice') or doc.metadata.get('price', '0'),
                    'hprice': doc.metadata.get('hprice', '0'),
                    'mallName': doc.metadata.get('mall', ''),
                    'image': doc.metadata.get('image', 'https://placehold.co/150'),
                    'brand': doc.metadata.get('brand', ''),
                    'maker': doc.metadata.get('maker', ''),
                    'category1': doc.metadata.get('category1', ''),
                    'category2': doc.metadata.get('category2', '')
                }
                
                # 색상 필터링: title에 색상 키워드가 포함된 경우 우선
                if color_keywords:
                    title_lower = product['title'].lower()
                    # 색상 키워드가 title에 있으면 추가
                    if any(color in title_lower for colors in [['파란', 'blue', 'navy'], ['하얀', 'white', 'ivory'], ['검', 'black'], ['빨', 'red']] for color in colors):
                        products.append(product)
                elif len(products) < VECTOR_SEARCH_K:
                    # 색상 필터 없으면 모든 결과 추가
                    products.append(product)
        
        # 색상 필터링 후 결과가 적으면 색상 필터 없이 추가
        if len(products) < 3 and color_keywords:
            print("색상 필터링 결과가 적어 추가 결과를 포함합니다.")
            for doc in docs:
                if doc.metadata and len(products) < VECTOR_SEARCH_K:
                    product = {
                        'title': doc.metadata.get('title', ''),
                        'link': doc.metadata.get('link', ''),
                        'lprice': doc.metadata.get('lprice') or doc.metadata.get('price', '0'),
                        'hprice': doc.metadata.get('hprice', '0'),
                        'mallName': doc.metadata.get('mall', ''),
                        'image': doc.metadata.get('image', 'https://placehold.co/150'),
                        'brand': doc.metadata.get('brand', ''),
                        'maker': doc.metadata.get('maker', ''),
                        'category1': doc.metadata.get('category1', ''),
                        'category2': doc.metadata.get('category2', '')
                    }
                    # 중복 체크
                    if not any(p.get('productId') == product.get('productId') for p in products):
                        products.append(product)

    # 6. 실습과제 2: 예산 조건이 있으면 검색 결과에 필터 및 정렬 적용
    if budget_info["condition"] and products:
        products = apply_budget_filter(products, budget_info)
        print(f"필터링 완료: {len(products)}개의 상품이 조건에 맞습니다.")

    # 7. 필터링된 결과(products)를 사용하여 응답 및 HTML 생성
    html_output = format_product_html(products)
    search_results = format_product_list(products) if products else None
    
    # LLM 응답 생성
    response = generate_response_with_context(
        message, 
        conversation_history, 
        search_results
    )
    
    # 8. 응답을 대화 히스토리에 추가 및 토큰 통계 관리
    conversation_history, _ = token_manager.manage_conversation_history(
        conversation_history,
        {"role": "assistant", "content": response}
    )
    
    token_stats = token_manager.get_token_stats(conversation_history)
    response += f"\n\n(Tip. '최신', '신상', '재고' 키워드를 포함해보세요.)"
    response += f"\n\n[검색: {search_count}/{MAX_SEARCH_COUNT}] [토큰: {token_stats['total']}/{MAX_CONTEXT_TOKENS}] [메시지: {token_stats['messages']}]"
    
    # 9. Gradio 형식으로 최종 대화 히스토리 업데이트 및 반환
    if history is None:
        history = []
    history.append([message, response])
    
    return history, html_output

def create_interface():
    """
    Gradio 인터페이스를 생성합니다.
    
    Returns:
        demo: Gradio Blocks 인터페이스
    """
    with gr.Blocks(title="AI 쇼핑 어시스턴트") as demo:
        gr.Markdown("# AI 쇼핑 어시스턴트")
        gr.Markdown("패션 이미지를 업로드하고 AI와 대화하며 쇼핑을 즐겨보세요!")
        
        with gr.Row():
            # 왼쪽 열: 이미지 업로드 및 탐지
            with gr.Column(scale=1):
                image_input = gr.Image(label="상품 이미지", type="pil")
                detect_btn = gr.Button("상품 탐지", variant="primary")
                output_image = gr.Image(label="탐지 결과")
                
            # 오른쪽 열: 채팅 인터페이스
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(height=500)
                msg = gr.Textbox(
                    label="메시지",
                    placeholder="상품에 대해 궁금한 점을 물어보세요...",
                    lines=2
                )
                with gr.Row():
                    submit = gr.Button("전송", variant="primary")
                    clear = gr.Button("초기화")
        
        # 두번째 행 탐지 및 추천 상품 목록 표시
        with gr.Row():
            detection_info = gr.HTML(label="탐지 정보")
        
        # 이벤트 핸들러 연결
        def handle_detection_and_chat(image):
            """이미지 탐지 후 채팅 업데이트"""
            img, html = detect_fashion_objects(image)
            # 대화 히스토리에서 마지막 어시스턴트 메시지 가져오기
            if conversation_history and conversation_history[-1]['role'] == 'assistant':
                chat_history = [[None, conversation_history[-1]['content']]]
            else:
                chat_history = []
            return img, html, chat_history
        
        detect_btn.click(
            fn=handle_detection_and_chat,
            inputs=image_input,
            outputs=[output_image, detection_info, chatbot]
        )
        
        # 메시지 전송 이벤트
        msg.submit(fn=chat_response_with_budget, inputs=[msg, chatbot], outputs=[chatbot, detection_info]).then(
            fn=lambda: "", outputs=msg
        )
        submit.click(fn=chat_response_with_budget, inputs=[msg, chatbot], outputs=[chatbot, detection_info]).then(
            fn=lambda: "", outputs=msg
        )
        
        # 대화 초기화 함수
        def clear_conversation():
            global conversation_history, search_count
            conversation_history = []
            search_count = 0
            return None, ""  # chatbot과 msg 둘 다 초기화
        
        clear.click(clear_conversation, outputs=[chatbot, msg])
    
    return demo

def load_initial_products():
    """
    fashion_products.json 파일에서 초기 상품 데이터를 벡터 스토어에 로드합니다.
    카테고리 균형을 맞춰 다양한 상품이 로드되도록 합니다.
    """
    import json
    
    json_file = 'fashion_products.json'
    if not os.path.exists(json_file):
        print(f"{json_file} 파일을 찾을 수 없습니다.")
        return
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            products = json.load(f)
        
        # 카테고리별 상품 수 제한 (균형 있는 데이터셋 구성)
        category_counts = {}
        max_pants = 30  # 바지는 최대 30개
        max_per_category = 60  # 다른 카테고리는 최대 60개
        
        # JSON 데이터를 네이버 API 형식으로 변환
        formatted_products = []
        for p in products:
            category = p.get('category', '')
            
            # 카테고리별 개수 체크
            if '바지' in category or 'pants' in category.lower():
                if category_counts.get('바지', 0) >= max_pants:
                    continue
                category_counts['바지'] = category_counts.get('바지', 0) + 1
            else:
                if category_counts.get(category, 0) >= max_per_category:
                    continue
                category_counts[category] = category_counts.get(category, 0) + 1
            
            formatted_products.append({
                'title': p.get('title', ''),
                'link': p.get('link', ''),
                'lprice': p.get('price', '0'),  # price를 lprice로 변환
                'hprice': p.get('price', '0'),
                'mallName': p.get('mall', ''),
                'image': p.get('image', 'https://placehold.co/150'),
                'productId': p.get('productId', ''),
                'brand': p.get('brand', ''),
                'maker': p.get('brand', ''),
                'category1': p.get('category', ''),
                'category2': ''
            })
            
            # 300개까지만 로드
            if len(formatted_products) >= 300:
                break
        
        print(f"카테고리별 분포: {category_counts}")
        print(f"{len(formatted_products)}개의 초기 상품 데이터를 벡터 스토어에 로딩 중...")
        save_products_to_vectorstore(formatted_products)
        
        # CLIP 이미지 벡터도 저장 (시간이 걸릴 수 있음)
        print("초기 상품의 이미지 벡터 추출 중... (시간이 걸릴 수 있습니다)")
        save_products_with_clip_vectors(formatted_products)
        
        print("초기 상품 데이터 로딩 완료!")
    except Exception as e:
        print(f"초기 데이터 로딩 오류: {e}")

if __name__ == "__main__":
    print("AI 쇼핑 어시스턴트 시작...")
    print("모델 로딩 중...")
    load_models()
    
    # 초기 상품 데이터 로드
    print("초기 상품 데이터 로딩 중...")
    load_initial_products()
    
    print("서버 시작...")
    demo = create_interface()
    demo.launch(
        server_port=GRADIO_SERVER_PORT,
        server_name=GRADIO_SERVER_NAME,
        share=False  # 공개 URL 생성 여부
    )