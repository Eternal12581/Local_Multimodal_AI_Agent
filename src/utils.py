"""通用工具函数模块"""
import os
import shutil
import logging
from pathlib import Path
from datetime import datetime
from config import *

# 配置日志
def setup_logger():
    """设置日志配置"""
    logger = logging.getLogger("local_ai_agent")
    logger.setLevel(LOG_LEVEL)
    
    # 确保日志目录存在
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 创建文件处理器和控制台处理器
    log_file = LOG_DIR / f"agent_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 初始化日志
logger = setup_logger()

def is_pdf_file(file_path):
    """检查文件是否为PDF"""
    return str(file_path).lower().endswith('.pdf')

def is_image_file(file_path):
    """检查文件是否为图像"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    ext = os.path.splitext(str(file_path).lower())[1]
    return ext in image_extensions

def create_topic_directories(base_dir, topics):
    """为每个主题创建目录"""
    for topic in topics:
        topic_dir = base_dir / topic
        os.makedirs(topic_dir, exist_ok=True)

def move_file_to_topic(file_path, base_dir, topic):
    """将文件移动到对应主题目录"""
    file_path = Path(file_path)
    target_dir = base_dir / topic
    os.makedirs(target_dir, exist_ok=True)
    
    # 处理同名文件
    target_path = target_dir / file_path.name
    counter = 1
    while target_path.exists():
        name, ext = os.path.splitext(file_path.name)
        target_path = target_dir / f"{name}_{counter}{ext}"
        counter += 1
    
    shutil.move(str(file_path), str(target_path))
    return target_path

def get_all_files_in_directory(directory, file_check_func):
    """获取目录中所有符合条件的文件"""
    files = []
    for root, _, file_names in os.walk(directory):
        for file_name in file_names:
            file_path = Path(root) / file_name
            if file_check_func(file_path):
                files.append(file_path)
    return files