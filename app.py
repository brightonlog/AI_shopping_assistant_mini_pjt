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

# 전역 변수
models = {}
vector_store = None
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
    
    print("모든 모델 로딩 완료!")
    return models

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
    업로드된 이미지에서 패션 아이템을 탐지합니다.
    
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
            
            # 탐지된 아이템으로 초기 상품 검색
            products = search_products(' '.join(detected_items[:2]))
            save_products_to_vectorstore(products)
            
            # HTML 출력 생성
            html_output = f"<h3>탐지된 아이템</h3><p>{', '.join(detected_items)}</p><hr>"
            html_output += format_product_html(products)
            
            # LLM 초기 메시지 생성
            detected_items_str = ', '.join(detected_items)
            
            try:
                # 간단한 초기 인사 메시지 생성
                llm_response = f"안녕하세요! 이미지에서 {detected_items_str}을(를) 발견했습니다. 어떤 스타일이나 브랜드를 선호하시나요? 예산도 알려주시면 더 정확한 추천을 도와드릴 수 있어요."
                
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
                'lprice': doc.metadata.get('price', '0'),
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

def chat_response(message, history):
    """
    사용자 메시지에 대한 챗봇 응답을 생성합니다.
    
    Args:
        message: 사용자 메시지
        history: Gradio 대화 히스토리
        
    Returns:
        history: 업데이트된 대화 히스토리
    """
    global search_count, conversation_history
    
    # 메시지가 비어있으면 현재 히스토리 반환
    if not message or not message.strip():
        return history
    
    # Gradio 히스토리를 내부 형식으로 변환
    if history:
        for h in history:
            if isinstance(h, list) and len(h) == 2:
                user_msg_content = h[0]
                assistant_msg_content = h[1]
                
                # None이나 빈 메시지 건너뛰기
                if not user_msg_content or not assistant_msg_content:
                    continue
                    
                # 문자열로 변환
                user_msg_content = str(user_msg_content) if user_msg_content else ""
                assistant_msg_content = str(assistant_msg_content) if assistant_msg_content else ""
                
                # 이미 추가된 대화는 스킵
                user_msg = {"role": "user", "content": user_msg_content}
                assistant_msg = {"role": "assistant", "content": assistant_msg_content}
                
                # 중복 확인 (content만 비교)
                user_exists = any(msg.get('content') == user_msg_content for msg in conversation_history if msg.get('role') == 'user')
                assistant_exists = any(msg.get('content') == assistant_msg_content for msg in conversation_history if msg.get('role') == 'assistant')
                
                if not user_exists and user_msg_content:
                    conversation_history.append(user_msg)
                if not assistant_exists and assistant_msg_content:
                    conversation_history.append(assistant_msg)
    
    # 현재 메시지 추가 및 토큰 관리
    conversation_history, token_count = token_manager.manage_conversation_history(
        conversation_history,
        {"role": "user", "content": message}
    )
    
    html_output = ''
    # 웹 검색 필요 여부 판단
    if should_search_web(message) and search_count < MAX_SEARCH_COUNT:
        # 웹에서 새로운 상품 검색
        print(f'\n웹 검색 : {message}\n')
        products = search_products(message)
        save_products_to_vectorstore(products)
        search_results = format_product_list(products)
        html_output = format_product_html(products)
        
        # LLM에 대화 컨텍스트와 함께 전달
        response = generate_response_with_context(
            message, 
            conversation_history, 
            search_results
        )
        
        # 검색 결과 추가
        if products:
            response += f"\n\n{search_results}"
    else:
        # 벡터 DB 에서 새로운 상품 검색
        print(f'\n벡터 DB 검색 : {message}\n')
        # 벡터 스토어에서 검색
        docs = vector_store.similarity_search(message, k=VECTOR_SEARCH_K)
        products = []
        
        for doc in docs:
            if doc.metadata:
                # 메타데이터에서 상품 정보 복원
                product = {
                    'title': doc.metadata.get('title', ''),
                    'link': doc.metadata.get('link', ''),
                    'lprice': doc.metadata.get('price', '0'),
                    'hprice': doc.metadata.get('hprice', '0'),
                    'mallName': doc.metadata.get('mall', ''),
                    'image': doc.metadata.get('image', 'https://placehold.co/150'),
                    'brand': doc.metadata.get('brand', ''),
                    'maker': doc.metadata.get('maker', ''),
                    'category1': doc.metadata.get('category1', ''),
                    'category2': doc.metadata.get('category2', '')
                }
                products.append(product)
        
        html_output = format_product_html(products)
        if products:
            search_results = format_product_list(products)
            response = generate_response_with_context(
                message,
                conversation_history,
                search_results
            )
        else:
            response = generate_response_with_context(
                message,
                conversation_history
            )
    
    # 응답을 대화 히스토리에 추가
    conversation_history, _ = token_manager.manage_conversation_history(
        conversation_history,
        {"role": "assistant", "content": response}
    )
    
    # 토큰 통계 정보 추가
    token_stats = token_manager.get_token_stats(conversation_history)
    response += f"\n\n(Tip. '최신', '신상', '재고', '실시간', '현재', '오늘'  키워드를 포함해보세요.)"
    response += f"\n\n[검색: {search_count}/{MAX_SEARCH_COUNT}] [토큰: {token_stats['total']}/{MAX_CONTEXT_TOKENS}] [메시지: {token_stats['messages']}]"
    
    # Gradio 형식으로 대화 히스토리 업데이트
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
        msg.submit(fn=chat_response, inputs=[msg, chatbot], outputs=[chatbot, detection_info]).then(
            fn=lambda: "", outputs=msg
        )
        submit.click(fn=chat_response, inputs=[msg, chatbot], outputs=[chatbot, detection_info]).then(
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

if __name__ == "__main__":
    print("AI 쇼핑 어시스턴트 시작...")
    print("모델 로딩 중...")
    load_models()
    print("서버 시작...")
    
    demo = create_interface()
    demo.launch(
        server_port=GRADIO_SERVER_PORT,
        server_name=GRADIO_SERVER_NAME,
        share=False  # 공개 URL 생성 여부
    )