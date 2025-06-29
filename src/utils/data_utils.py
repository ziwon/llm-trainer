import json
from pathlib import Path
from typing import Dict, List, Any
from datasets import Dataset, load_dataset

class DataProcessor:
    """데이터 처리를 위한 유틸리티 클래스"""
    
    @staticmethod
    def load_dataset(dataset_name: str) -> Dataset:
        """데이터셋 로드"""
        if dataset_name.endswith('.json'):
            with open(dataset_name, 'r') as f:
                data = json.load(f)
            return Dataset.from_list(data)
        else:
            return load_dataset(dataset_name, split="train")
    
    @staticmethod
    def format_alpaca_prompt(instruction: str, input_text: str = "", output: str = "") -> str:
        """Alpaca 형식 프롬프트 생성"""
        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    @staticmethod
    def create_sample_dataset(output_path: str, num_samples: int = 100):
        """샘플 데이터셋 생성"""
        sample_data = []
        for i in range(num_samples):
            sample_data.append({
                "instruction": f"샘플 지시사항 {i+1}",
                "input": f"샘플 입력 {i+1}" if i % 2 == 0 else "",
                "output": f"샘플 출력 {i+1}"
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)