# utils.py
"""
AI 쇼핑 어시스턴트 유틸리티 모듈
토큰 관리 및 기타 유틸리티 함수들
"""

import torch
from typing import List, Dict, Tuple


class TokenManager:
    """
    대화 히스토리의 토큰을 관리하는 클래스
    토큰 수를 추적하고 컨텍스트 윈도우 내에서 대화를 유지합니다.
    """
    
    def __init__(self, tokenizer, max_context_tokens=2048):
        """
        TokenManager 초기화
        
        Args:
            tokenizer: Hugging Face tokenizer 객체
            max_context_tokens: 최대 컨텍스트 토큰 수
        """
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens
        
    def count_tokens(self, text: str) -> int:
        """
        텍스트의 토큰 수를 계산합니다.
        
        Args:
            text: 토큰을 계산할 텍스트
            
        Returns:
            int: 토큰 수
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    
    def prepare_prompt(self, system_prompt: str, conversation_history: List[Dict], 
                      current_query: str, context: str = None) -> Tuple[str, int]:
        """
        대화 프롬프트를 준비하고 토큰 수를 계산합니다.
        
        Args:
            system_prompt: 시스템 프롬프트
            conversation_history: 대화 히스토리
            current_query: 현재 사용자 쿼리
            context: 추가 컨텍스트 (검색 결과 등)
            
        Returns:
            Tuple[str, int]: (준비된 프롬프트, 토큰 수)
        """
        # 프롬프트 구성
        prompt_parts = [f"시스템: {system_prompt}\n"]
        
        # 대화 히스토리 추가
        for msg in conversation_history:
            role = "사용자" if msg["role"] == "user" else "어시스턴트"
            prompt_parts.append(f"{role}: {msg['content']}\n")
        
        # 컨텍스트가 있으면 추가
        if context:
            prompt_parts.append(f"검색 결과:\n{context}\n")
        
        # 현재 쿼리 추가
        prompt_parts.append(f"사용자: {current_query}\n")
        prompt_parts.append("어시스턴트:")
        
        full_prompt = "".join(prompt_parts)
        token_count = self.count_tokens(full_prompt)
        
        return full_prompt, token_count
    
    def manage_conversation_history(self, history: List[Dict], 
                                  new_message: Dict) -> Tuple[List[Dict], int]:
        """
        대화 히스토리를 관리하고 토큰 한계 내에서 유지합니다.
        
        Args:
            history: 현재 대화 히스토리
            new_message: 추가할 새 메시지
            
        Returns:
            Tuple[List[Dict], int]: (업데이트된 히스토리, 총 토큰 수)
        """
        # 새 메시지 추가
        updated_history = history + [new_message]
        
        # 토큰 수 계산
        total_tokens = sum(self.count_tokens(msg['content']) for msg in updated_history)
        
        # 토큰 한계 초과 시 오래된 메시지 제거
        while total_tokens > self.max_context_tokens and len(updated_history) > 2:
            # 가장 오래된 사용자-어시스턴트 쌍 제거
            if updated_history[0]['role'] == 'user' and len(updated_history) > 1:
                updated_history = updated_history[2:]
            else:
                updated_history = updated_history[1:]
            
            # 토큰 수 재계산
            total_tokens = sum(self.count_tokens(msg['content']) for msg in updated_history)
        
        return updated_history, total_tokens
    
    def get_token_stats(self, history: List[Dict]) -> Dict[str, int]:
        """
        대화 히스토리의 토큰 통계를 계산합니다.
        
        Args:
            history: 대화 히스토리
            
        Returns:
            Dict[str, int]: 토큰 통계 정보
        """
        total_tokens = sum(self.count_tokens(msg['content']) for msg in history)
        message_count = len(history)
        
        return {
            'total': total_tokens,
            'messages': message_count,
            'average': total_tokens // message_count if message_count > 0 else 0,
            'remaining': self.max_context_tokens - total_tokens
        }
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        """
        텍스트를 최대 토큰 수에 맞게 잘라냅니다.
        
        Args:
            text: 자를 텍스트
            max_tokens: 최대 토큰 수
            
        Returns:
            str: 잘린 텍스트
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_tokens:
            return text
        
        # 토큰을 자르고 디코드
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        return truncated_text + "..."


def clean_text(text: str) -> str:
    """
    텍스트를 정리합니다.
    
    Args:
        text: 정리할 텍스트
        
    Returns:
        str: 정리된 텍스트
    """
    # 여러 개의 공백을 하나로
    text = ' '.join(text.split())
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    return text


def format_price(price: str) -> str:
    """
    가격을 포맷팅합니다.
    
    Args:
        price: 가격 문자열
        
    Returns:
        str: 포맷팅된 가격
    """
    try:
        price_int = int(price)
        return f"{price_int:,}원"
    except:
        return price