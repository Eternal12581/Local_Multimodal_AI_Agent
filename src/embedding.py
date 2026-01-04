"""嵌入生成模块（本地静态模型版：CLIP+文本嵌入模型+本地LLM，显式指定GPU+显存优化+CPU兜底）"""
import torch
import os
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, AutoModelForCausalLM, AutoTokenizer
import config
from config import *
import logging

TARGET_GPU_ID = 0

os.environ["CUDA_VISIBLE_DEVICES"] = str(TARGET_GPU_ID)

gpu_logger = logging.getLogger("local_ai_agent")
if torch.cuda.is_available():
    available_gpu_count = torch.cuda.device_count()
    current_gpu_name = torch.cuda.get_device_name(0)
else:
    gpu_logger.warning("⚠️  指定的GPU不可用或CUDA未配置，将自动切换到CPU运行")
# ==============================================================================================================

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 初始化正式日志
logger = logging.getLogger("local_ai_agent")

def clear_gpu_cache():
    """清空GPU显存缓存，释放无用占用"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.debug("GPU显存缓存已清空")

class ImageEmbedding:
    """使用本地静态CLIP模型进行图像/文本嵌入"""
    def __init__(self):
        clear_gpu_cache()
        self.device = torch.device("cuda:0") if (torch.cuda.is_available()) else torch.device("cpu")
        if self.device.type == "cpu":
            logger.warning("⚠️  GPU不可用或显存不足，自动切换到CPU运行")
        
        # 本地CLIP模型路径
        self.clip_model_path = Path(IMAGE_EMBEDDING_MODEL).absolute()
        
        if not self.clip_model_path.exists():
            raise FileNotFoundError(f"本地CLIP模型路径不存在：{self.clip_model_path}")
        
        try:
            logger.info(f"加载本地静态CLIP模型: {self.clip_model_path}")
            
            self.processor = CLIPProcessor.from_pretrained(
                str(self.clip_model_path),
                local_files_only=True,
                cache_dir=str(IMAGE_EMBEDDING_CACHE_DIR),
                trust_remote_code=True,
                use_fast=False
            )
            
            # 加载CLIP Model
            dtype = TORCH_DTYPE if self.device.type == "cuda" else torch.float32
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
                dtype = torch.float16

            self.model = CLIPModel.from_pretrained(
                str(self.clip_model_path),
                local_files_only=True,
                cache_dir=str(IMAGE_EMBEDDING_CACHE_DIR),
                dtype=dtype,
                trust_remote_code=True
            ).to(self.device)
            
            self.model.eval()
            
            clear_gpu_cache()
            logger.info("✅ 本地CLIP模型加载成功")
        
        except Exception as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower():
                logger.error("❌ CLIP模型加载时GPU显存不足，尝试切换到CPU运行...")
                dtype = torch.float32
                self.model = CLIPModel.from_pretrained(
                    str(self.clip_model_path),
                    local_files_only=True,
                    cache_dir=str(IMAGE_EMBEDDING_CACHE_DIR),
                    dtype=dtype,
                    trust_remote_code=True
                ).to(self.device)
                self.model.eval()
                clear_gpu_cache()
                logger.info("✅ CLIP模型已切换到CPU加载成功")
            else:
                logger.error(f"❌ 加载本地CLIP模型失败: {error_msg}")
                raise
    
    def embed(self, image_paths):
        """生成图像嵌入向量"""
        images = []
        for img_path in image_paths:
            try:
                img = Image.open(str(img_path)).convert("RGB")
                images.append(img)
            except Exception as e:
                logger.error(f"❌ 加载图像 {img_path} 失败: {str(e)}")
                images.append(None)
        
        # 过滤无效图像
        valid_images = [img for img in images if img is not None]
        if not valid_images:
            return np.array([])
        
        try:
            inputs = self.processor(
                images=valid_images,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device) 
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                embeddings = image_features.cpu().numpy()
            
            # 归一化
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            
            # 清空显存缓存
            clear_gpu_cache()
            return embeddings / norms
        
        except Exception as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower():
                logger.error("❌ 图像嵌入生成时GPU显存不足，已切换到CPU")
            else:
                logger.error(f"❌ 生成图像嵌入失败: {error_msg}")
            return np.zeros((len(valid_images), IMAGE_EMBEDDING_DIM))
    
    def embed_text(self, texts):
        """生成文本嵌入向量"""
        if not texts or not isinstance(texts, list):
            return np.array([])
        
        try:
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                embeddings = text_features.cpu().numpy()
            
            # 归一化
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            
            # 清空显存缓存
            clear_gpu_cache()
            return embeddings / norms
        
        except Exception as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower():
                logger.error("❌ 文本嵌入生成时GPU显存不足，已切换到CPU")
            else:
                logger.error(f"❌ 生成文本嵌入失败: {error_msg}")
            # 兜底返回零向量
            return np.zeros((len(texts), IMAGE_EMBEDDING_DIM))

class TextEmbedding:
    """使用本地静态文本嵌入模型+本地LLM"""
    def __init__(self):
    
        clear_gpu_cache()
        
        # 1. 初始化本地文本嵌入模型
        self.embed_device = torch.device("cuda:0") if (torch.cuda.is_available()) else torch.device("cpu")
        if self.embed_device.type == "cpu":
            logger.warning("⚠️  GPU不可用或显存不足，文本嵌入模型自动切换到CPU运行")
        
        self.embed_model_path = Path(TEXT_EMBEDDING_MODEL).absolute()
        
        if not self.embed_model_path.exists():
            raise FileNotFoundError(f"本地文本嵌入模型路径不存在：{self.embed_model_path}")
        
        try:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                logger.error("❌ 未安装sentence-transformers，无法加载文本嵌入模型，请执行 pip install sentence-transformers 安装")
                raise
            
            # 加载文本嵌入模型
            self.embed_model = SentenceTransformer(
                str(self.embed_model_path),
                device=str(self.embed_device),
                cache_folder=str(TEXT_EMBEDDING_CACHE_DIR)
            )
            
            self.embed_batch_size = max(1, TEXT_EMBEDDING_BATCH_SIZE // 2)
            logger.info(f"✅ 本地文本嵌入模型加载成功：{self.embed_model_path}（批处理大小优化为：{self.embed_batch_size}）")
            
            # 清空显存缓存
            clear_gpu_cache()
        except Exception as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower():
                logger.error("❌ 文本嵌入模型加载时GPU显存不足，尝试切换到CPU运行...")
                from sentence_transformers import SentenceTransformer
                self.embed_model = SentenceTransformer(
                    str(self.embed_model_path),
                    device="cpu",
                    cache_folder=str(TEXT_EMBEDDING_CACHE_DIR)
                )
                self.embed_batch_size = 1
                clear_gpu_cache()
                logger.info("✅ 文本嵌入模型已切换到CPU加载成功")
            else:
                logger.error(f"❌ 加载本地文本嵌入模型失败: {str(e)}")
                raise
        
        # 2. 初始化本地LLM模型
        self.llm_available = False
        self.llm_tokenizer = None
        self.llm_model = None
        
        if config.USE_LLM_FOR_QUERY_UNDERSTANDING:
            config.USE_LLM_FOR_CLASSIFICATION = True
            config.USE_LLM_FOR_RERANKING = True
            
            self.llm_model_path = Path(LOCAL_LLM_MODEL).absolute()
            
            if not self.llm_model_path.exists():
                raise FileNotFoundError(f"本地LLM模型路径不存在：{self.llm_model_path}")
            
            try:
                logger.info(f"加载本地LLM模型: {self.llm_model_path}")
                
                self.llm_tokenizer = AutoTokenizer.from_pretrained(
                    str(self.llm_model_path),
                    local_files_only=True,
                    cache_dir=str(os.path.join(DATA_DIR, "model_cache", "llm")),
                    trust_remote_code=True
                )
                
                llm_device = "cuda:0" if torch.cuda.is_available() else "cpu"
                llm_dtype = LOCAL_LLM_DTYPE if llm_device == "cuda:0" else torch.float32
                if llm_device == "cuda:0" and torch.cuda.is_bf16_supported():
                    llm_dtype = torch.float16
    
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    str(self.llm_model_path),
                    local_files_only=True,
                    cache_dir=str(os.path.join(DATA_DIR, "model_cache", "llm")),
                    device_map="auto" if llm_device == "cuda:0" else llm_device,
                    dtype=llm_dtype,  
                    max_memory={0: MAX_LLM_GPU_MEMORY} if llm_device == "cuda:0" else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True 
                )
                
                self.llm_model.eval()
                
                clear_gpu_cache()
                self.llm_available = True
                logger.info(f"✅ 本地LLM模型加载成功")
            except Exception as e:
                error_msg = str(e)
                if "out of memory" in error_msg.lower():
                    logger.error("❌ LLM模型加载时GPU显存不足，将使用基础文本嵌入功能")
                else:
                    logger.error(f"❌ 加载本地LLM模型失败: {str(e)}")
                config.USE_LLM_FOR_QUERY_UNDERSTANDING = False
                config.USE_LLM_FOR_CLASSIFICATION = False
                config.USE_LLM_FOR_RERANKING = False
                clear_gpu_cache()
                logger.warning("⚠️  所有LLM功能已自动关闭，将使用基础文本嵌入功能")
        else:
            config.USE_LLM_FOR_CLASSIFICATION = False
            config.USE_LLM_FOR_RERANKING = False
            logger.info("ℹ️  LLM所有活动已禁用（总开关关闭），未加载本地LLM模型，无额外显存/磁盘占用")
    # ==============================================================================================================
    
    def embed(self, texts):
        """生成文本嵌入向量"""
        if not texts or not isinstance(texts, list):
            return np.array([])
        
        try:
            # 本地模型推理
            embeddings = self.embed_model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=self.embed_batch_size
            )
            
            clear_gpu_cache()
            return embeddings
        except Exception as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower():
                logger.error("❌ 文本嵌入生成时GPU显存不足，已切换到CPU")
            else:
                logger.error(f"❌ 生成文本嵌入失败: {str(e)}")
            return np.zeros((len(texts), TEXT_EMBEDDING_DIM))
    
    def generate_text(self, prompt, max_tokens=None):
        """调用本地LLM生成文本"""
        max_tokens = max_tokens or LOCAL_LLM_MAX_TOKENS
        temperature = LOCAL_LLM_TEMPERATURE
        
        if not self.llm_available or not config.USE_LLM_FOR_QUERY_UNDERSTANDING or not prompt or not isinstance(prompt, str):
            logger.warning("⚠️  本地LLM不可用，无法生成文本")
            return ""
        
        try:
            max_prompt_length = LOCAL_LLM_MAX_CONTEXT_LENGTH - max_tokens - 10
            prompt = prompt[:max_prompt_length].strip()

            inputs = self.llm_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=LOCAL_LLM_MAX_CONTEXT_LENGTH
            ).to(next(self.llm_model.parameters()).device)
            
            from transformers import GenerationConfig
            generation_config = GenerationConfig(
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=(temperature > 0.0),
                pad_token_id=self.llm_tokenizer.eos_token_id,
                eos_token_id=self.llm_tokenizer.eos_token_id,
                num_beams=1 if temperature == 0.0 else 4,
                use_cache=True
            )

            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    generation_config=generation_config
                )

            full_text = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = full_text[len(prompt):].strip()

            clear_gpu_cache()
            logger.debug(f"✅ 本地LLM文本生成成功（生成长度：{len(generated_text)}字符）")
            return generated_text
        
        except Exception as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower():
                logger.error("❌ LLM文本生成时GPU显存不足，返回空字符串")
            else:
                logger.error(f"❌ 本地LLM文本生成失败: {str(e)}")
            # 清空显存缓存
            clear_gpu_cache()
            return ""