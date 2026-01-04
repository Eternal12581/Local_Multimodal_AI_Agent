"""
配置文件：存储项目所需的各类配置参数
"""

import os
import torch
from pathlib import Path

# 自动检测GPU设备
def get_available_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available(): 
        return "mps"
    else:
        return "cpu"

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.resolve()

# 数据存储路径配置
DATA_DIR = PROJECT_ROOT / "data"
os.makedirs(DATA_DIR, exist_ok=True)

# 文献存储相关路径
PAPERS_DIR = DATA_DIR / "papers"
os.makedirs(PAPERS_DIR, exist_ok=True)

# 图像存储相关路径
IMAGES_DIR = DATA_DIR / "images"
os.makedirs(IMAGES_DIR, exist_ok=True)

# 向量数据库配置
VECTOR_DB_DIR = DATA_DIR / "vector_db"
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
VECTOR_DB_PAPERS_COLLECTION = "papers"
VECTOR_DB_IMAGES_COLLECTION = "images"
VECTOR_DB_PERSIST_DIRECTORY = VECTOR_DB_DIR / "persist"
os.makedirs(VECTOR_DB_PERSIST_DIRECTORY, exist_ok=True)

# 设备配置（自动检测GPU）
DEVICE = get_available_device()

# ===================== 本地静态模型配置 =====================
# 本地模型根目录
LOCAL_MODEL_DIR = PROJECT_ROOT / "models"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# 1. 文本嵌入模型
TEXT_EMBEDDING_LOCAL_PATH = str(LOCAL_MODEL_DIR / "all-MiniLM-L6-v2")
TEXT_EMBEDDING_MODEL = TEXT_EMBEDDING_LOCAL_PATH
TEXT_EMBEDDING_MODEL_DEVICE = DEVICE
TEXT_EMBEDDING_BATCH_SIZE = 32 if DEVICE != "cpu" else 8
TEXT_EMBEDDING_CACHE_DIR = DATA_DIR / "model_cache" / "text_embedding"
os.makedirs(TEXT_EMBEDDING_CACHE_DIR, exist_ok=True)
TEXT_EMBEDDING_DIM = 384

# 2. 图像嵌入模型
IMAGE_EMBEDDING_LOCAL_PATH = str(LOCAL_MODEL_DIR / "clip-ViT-L-14/0_CLIPModel")
IMAGE_EMBEDDING_MODEL = IMAGE_EMBEDDING_LOCAL_PATH 
IMAGE_EMBEDDING_MODEL_DEVICE = DEVICE
IMAGE_EMBEDDING_BATCH_SIZE = 16 if DEVICE != "cpu" else 4
IMAGE_EMBEDDING_CACHE_DIR = DATA_DIR / "model_cache" / "image_embedding"
os.makedirs(IMAGE_EMBEDDING_CACHE_DIR, exist_ok=True)
IMAGE_EMBEDDING_DIM = 768

# 3. 本地LLM模型配置
# 本地LLM模型路径
LOCAL_LLM_MODEL_PATH = str(LOCAL_MODEL_DIR / "Qwen2-7B-Instruct")
LOCAL_LLM_MODEL = LOCAL_LLM_MODEL_PATH
LOCAL_LLM_DEVICE = DEVICE
LOCAL_LLM_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
LOCAL_LLM_MAX_CONTEXT_LENGTH = 2048  # LLM最大上下文长度
LOCAL_LLM_MAX_TOKENS = 512  # LLM生成文本的最大长度
LOCAL_LLM_TEMPERATURE = 0.0  # 分类/重排序用0.0（确定性输出），生成用0.7
# 本地LLM嵌入维度
LOCAL_LLM_EMBEDDING_DIM = 3584

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_CACHE"] = str(DATA_DIR / "model_cache" / "transformers") 
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
# ================================================================

# 分类
DEFAULT_TOPICS = [
    "Computer Vision, image processing, visual recognition, CVPR, visual model, convolutional neural network, CNN, image classification, object detection, VGG, ResNet, GoogLeNet, AlexNet",  # CV
    "Natural Language Processing, language model, NLP, text generation, LLM, transformer, BERT, GPT",    # NLP
    "Reinforcement Learning, RL, policy gradient, reward function, agent, Q-learning, PPO",       # RL
    "Multimodal, Vision-Language, CLIP, BLIP, LLaVA, cross-modal, image-text, video-text"                # 多模态
]

# 分类阈值配置
SIMILARITY_THRESHOLD = 0.25  # 嵌入分类阈值
USE_LLM_FOR_CLASSIFICATION = True  # 启用LLM分类

# 搜索相关配置
SEARCH_RESULTS_LIMIT = 20 if DEVICE != "cpu" else 10
SEARCH_SIMILARITY_THRESHOLD = 0.15
USE_LLM_FOR_QUERY_UNDERSTANDING = False  # 启用LLM查询理解
USE_LLM_FOR_RERANKING = True  # 启用LLM结果重排序
LLM_RERANK_TOP_K = 20  # LLM重排序候选数量
LLM_RERANK_BATCH_SIZE = 5  # 本地LLM批处理量

# PDF文本处理配置
MAX_PARAGRAPHS_PER_PDF = 20  # 单篇PDF提取的最大段落数
PARAGRAPH_CHUNK_SIZE = 1000   # 段落文本分块大小
PARAGRAPH_OVERLAP = 200       # 段落分块之间的重叠长度
SNIPPETS_TOP_K = 1            # 每个PDF返回的顶部匹配片段数

# 图像存储与搜索配置
IMAGES_DIR = "data/images"  # 图像存储目录
SEARCH_RESULTS_LIMIT = 10   # 搜索默认返回结果数量
SEARCH_SIMILARITY_THRESHOLD = 0.5  # 相似度阈值

# 日志配置
LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_LEVEL = "INFO"

# 段落级存储配置
PARAGRAPH_CHUNK_SIZE = 500  # 每个段落的最大字符数
PARAGRAPH_OVERLAP = 50  # 段落之间的重叠字符数

# GPU内存优化配置
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MAX_GPU_MEMORY = "8GB"  # 显存配置
MAX_LLM_GPU_MEMORY = "8GB"  # 本地LLM专属显存限制
# API配置
OPENAI_API_KEY = ""
GEMINI_API_KEY = ""