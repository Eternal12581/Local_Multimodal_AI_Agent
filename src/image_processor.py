"""图像处理模块，负责图像嵌入和存储"""
import os
import shutil
import hashlib
from pathlib import Path
from PIL import Image
from .embedding import ImageEmbedding
from .vector_db import VectorDB
from .utils import logger, is_image_file, get_all_files_in_directory
from config import *
import datetime

class ImageProcessor:
    def __init__(self):
        self.image_embedder = ImageEmbedding()
        self.vector_db = VectorDB()
        
        self.images_dir = Path(IMAGES_DIR) if isinstance(IMAGES_DIR, str) else IMAGES_DIR
        self.images_dir.mkdir(parents=True, exist_ok=True)
    
    def get_image_id(self, file_path):
        """生成唯一的图像ID"""
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"文件不存在，无法生成图像ID: {file_path}")
            return None
        file_info = f"{str(file_path.absolute())}|{os.path.getmtime(str(file_path.absolute()))}"
        return hashlib.md5(file_info.encode('utf-8')).hexdigest()
    
    def process_single_image(self, image_path):
        """处理单张图像（核心逻辑）"""
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.error(f"图像文件不存在: {image_path.absolute()}")
            return None
            
        if not is_image_file(image_path):
            logger.error(f"不是图像文件: {image_path.absolute()}")
            return None
        
        # 生成图像嵌入
        logger.info(f"正在处理图像: {image_path.absolute()}")
        embeddings = self.image_embedder.embed([str(image_path.absolute())])
        
        if embeddings is None or len(embeddings) == 0:
            logger.error(f"生成图像嵌入失败: {image_path.absolute()}")
            return None
        
        # 生成图像ID
        image_id = self.get_image_id(image_path)
        if not image_id:
            logger.error(f"生成图像ID失败: {image_path.absolute()}")
            return None
        
        # 移动图像到存储目录
        target_path = self.images_dir / image_path.name
        counter = 1
        while target_path.exists():
            name, ext = os.path.splitext(image_path.name)
            target_path = self.images_dir / f"{name}_{counter}{ext}"
            counter += 1
        
        try:
            shutil.copy(str(image_path.absolute()), str(target_path.absolute()))
        except Exception as e:
            logger.error(f"复制图像文件失败: {str(e)}")
            return None
        
        # 存储到向量数据库
        metadata = {
            "path": str(target_path.absolute()),
            "filename": target_path.name,
            "added_at": datetime.datetime.now().isoformat()
        }
        
        try:
            self.vector_db.add_images(
                image_ids=[image_id],
                embeddings=embeddings,
                metadatas=[metadata]
            )
        except Exception as e:
            logger.error(f"存储图像到向量数据库失败: {str(e)}")
            # 清理已复制的无效文件
            if target_path.exists():
                os.remove(target_path)
            return None
        
        logger.info(f"图像处理完成: {target_path.absolute()}")
        return {
            "id": image_id,
            "path": str(target_path.absolute())
        }
    
    def add_image(self, image_path):
        """添加单张图像（兼容文件/目录，但提示批量命令）"""
        image_path = Path(image_path)
        if image_path.is_dir():
            logger.warning("传入的是目录路径，建议使用 batch_add_images 命令批量处理")
            return self.batch_add_images(image_path)
        
        return self.process_single_image(image_path)
    
    def batch_add_images(self, directory):
        """批量处理目录中的所有图像"""
        try:
            from tqdm import tqdm
        except ImportError:
            logger.warning("未安装tqdm，批量处理不显示进度条，可执行 pip install tqdm 安装")
            def tqdm(iterable, desc=None, unit=None):
                return iterable
        
        directory = Path(directory)
        if not directory.exists():
            logger.error(f"目录不存在: {directory.absolute()}")
            return []
            
        # 获取所有图像文件
        image_files = get_all_files_in_directory(directory, is_image_file)
        image_files = list(image_files) if image_files else []
        logger.info(f"发现 {len(image_files)} 个图像文件待处理")
        
        if not image_files:
            logger.warning("目录中没有找到图像文件")
            return []
        
        results = []
        # 使用进度条显示处理进度
        for img_path in tqdm(image_files, desc="处理图像", unit="张"):
            try:
                result = self.process_single_image(img_path)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"处理图像失败 {img_path.absolute()}: {str(e)}")
                continue
        
        logger.info(f"批量处理完成，共成功处理 {len(results)}/{len(image_files)} 个图像")
        return results
    
    def search_images(self, query, limit=SEARCH_RESULTS_LIMIT):
        """通过文本描述搜索图像（使用 CLIP，文本和图像在同一嵌入空间，无需投影）"""
        if not query or not isinstance(query, str):
            logger.warning("搜索查询为空或格式无效")
            return []
        
        # 使用 CLIP 的文本编码器直接生成查询嵌入
        text_embeddings = self.image_embedder.embed_text([query])
        if text_embeddings is None or len(text_embeddings) == 0:
            logger.warning("生成文本嵌入失败")
            return []
        
        query_embedding = text_embeddings[0].tolist()
        
        try:
            results = self.vector_db.search_images(query_embedding, limit * 2)
        except Exception as e:
            logger.error(f"向量数据库搜索失败: {str(e)}")
            return []
        
        # 整理结果并计算相似度
        search_results = []
        if not results or 'ids' not in results or len(results['ids']) == 0 or len(results['ids'][0]) == 0:
            return []
        
        for i in range(len(results['ids'][0])):
            try:
                distance = float(results['distances'][0][i])
                similarity = 1 - (distance / 2)
            except (IndexError, ValueError) as e:
                logger.error(f"解析相似度失败: {str(e)}")
                continue
            
            # 只保留相似度大于阈值的结果
            if similarity >= SEARCH_SIMILARITY_THRESHOLD:
                try:
                    metadata = results['metadatas'][0][i]
                    search_results.append({
                        "id": results['ids'][0][i],
                        "path": metadata.get('path', ''),
                        "filename": metadata.get('filename', ''),
                        "similarity": round(similarity, 4)
                    })
                except IndexError as e:
                    logger.error(f"解析图像元数据失败: {str(e)}")
                    continue
        
        # 按相似度降序排序
        search_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return search_results[:limit]
    
    # 格式化搜索结果
    def format_search_results(self, search_results):
        """
        格式化搜索结果输出，格式要求：
        1. mountain_view.jpg
           路径: data/images/mountain_view.jpg
           相似度: 0.9456
        """
        if not search_results:
            print("⚠️  未搜索到符合条件的图像")
            return
        
        for idx, result in enumerate(search_results, start=1):
            filename = result.get('filename', '未知文件名')
            path = result.get('path', '未知路径')
            similarity = result.get('similarity', 0.0)
            
            print(f"{idx}. {filename}")
            print(f"   路径: {path}")
            print(f"   相似度: {similarity:.4f}")
            print()
    
    def search_and_format(self, query, limit=SEARCH_RESULTS_LIMIT):
        """一键执行图像搜索并按指定格式输出结果"""
        search_results = self.search_images(query, limit)
        self.format_search_results(search_results)