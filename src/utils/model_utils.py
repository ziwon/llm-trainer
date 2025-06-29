import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

class ModelManager:
    """모델 관리를 위한 유틸리티 클래스"""
    
    @staticmethod
    def load_model_and_tokenizer(model_name: str, use_lora: bool = False, lora_config: dict = None):
        """모델과 토크나이저 로드"""
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=None,
            low_cpu_mem_usage=True
        )
        
        if use_lora and lora_config:
            model = ModelManager.apply_lora(model, lora_config)
        
        return model, tokenizer
    
    @staticmethod
    def apply_lora(model, lora_config: dict):
        """LoRA 적용"""
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("alpha", 32),
            lora_dropout=lora_config.get("dropout", 0.1),
            target_modules=lora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            bias="none"
        )
        return get_peft_model(model, config)
    
    @staticmethod
    def estimate_model_size(model):
        """모델 크기 추정"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb