# crawl_products.py
import requests
from bs4 import BeautifulSoup
import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import pandas as pd
from config import *

CRAWL_MAX_PAGES = 5  # 최대 몇 페이지를 가져올지 설정

def crawl_naver_shopping_api(query, max_pages=CRAWL_MAX_PAGES):
    client_id = 'QRVq6rn4UP6lhU4OlOdS'       # 네이버 개발자센터에서 받은 클라이언트 아이디
    client_secret = '8vBs5mOq1t' # 네이버 개발자센터에서 받은 클라이언트 시크릿
    
    products = []
    
    for page in range(1, max_pages + 1):
        start = (page - 1) * 20 + 1  # 네이버 API는 시작 위치가 1부터 시작, 20개씩
        
        url = "https://openapi.naver.com/v1/search/shop.json"
        headers = {
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret
        }
        params = {
            "query": query,
            "start": start,
            "display": 20,  # 한번에 20개씩 가져오기
            "sort": "sim"
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            break
        
        data = response.json()
        
        for item in data.get('items', []):
            product_type = int(item.get('productType', 0))
            if product_type in [1, 2, 3]:  # 일반상품만
                try:
                    # HTML 태그 제거 (제목에 <b> 태그가 포함될 수 있음)
                    title = item.get('title', '').replace('<b>', '').replace('</b>', '')
                    # 상품 정보 수집
                    product_info = {
                        'title': title,
                        'price': item.get('lprice', '0'),        # 최저가
                        'link': item.get('link', ''),
                        'mall': item.get('mallName', ''),
                        'image': item.get('image', ''),           # 이미지 URL 추가
                        'brand': item.get('brand', ''),           # 브랜드 정보 추가
                        'productId': item.get('productId', ''),   # 상품 ID 추가
                        'category': query
                    }
                    
                    products.append(product_info)
                except Exception as e:
                    print(f"Parsing error: {e}")
                    continue
        
        time.sleep(1)  # API 요청 과다 방지
        
    return products

def create_fashion_dataset():
    all_products = []
    
    for category in CRAWL_CATEGORIES:
        print(f"크롤링 중: {category}")
        products = crawl_naver_shopping_api(category)
        all_products.extend(products)
        time.sleep(2)
    
    # DataFrame으로 변환 및 저장
    df = pd.DataFrame(all_products)
    df.to_csv('fashion_products.csv', index=False, encoding='utf-8-sig')
    
    # JSON으로도 저장
    with open('fashion_products.json', 'w', encoding='utf-8') as f:
        json.dump(all_products, f, ensure_ascii=False, indent=2)
    
    print(f"총 {len(all_products)}개 상품 크롤링 완료")
    return df

def load_to_vectorstore(csv_path='fashion_products.csv'):
    from langchain_community.document_loaders import CSVLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    
    # CSV 로드
    loader = CSVLoader(csv_path, encoding='utf-8-sig')
    documents = loader.load()
    
    # 임베딩 및 벡터 스토어 생성
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH
    )
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    
    vectorstore.persist()
    print("벡터 스토어 생성 완료")

if __name__ == "__main__":
    # 1. 데이터 크롤링
    df = create_fashion_dataset()
    
    # 2. 벡터 스토어에 로드
    load_to_vectorstore()
    
    # 3. 통계 출력
    print("\n크롤링 결과:")
    print(f"총 상품 수: {len(df)}")
    print(f"카테고리별 상품 수:\n{df['category'].value_counts()}")
    print(f"평균 가격: {df['price'].astype(float).mean():,.0f}원")