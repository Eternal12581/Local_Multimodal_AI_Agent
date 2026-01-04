"""向量数据库操作模块，基于ChromaDB"""
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient
from config import *

class VectorDB:
    def __init__(self):
        
        self.client = PersistentClient(
            path=str(VECTOR_DB_PERSIST_DIRECTORY), 
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # 获取或创建文献集合
        self.papers_collection = self.client.get_or_create_collection(
            name=VECTOR_DB_PAPERS_COLLECTION,
            metadata={"hnsw:space": "cosine"} 
        )
        
        # 获取或创建图像集合
        self.images_collection = self.client.get_or_create_collection(
            name=VECTOR_DB_IMAGES_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
        
    def add_papers(self, paper_ids, embeddings, metadatas):
        """添加文献向量到数据库"""
        self.papers_collection.add(
            ids=paper_ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
    def add_images(self, image_ids, embeddings, metadatas):
        """添加图像向量到数据库"""
        self.images_collection.add(
            ids=image_ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
    def search_papers(self, query_embedding, limit=SEARCH_RESULTS_LIMIT):
        """搜索相关文献"""
        return self.papers_collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
        )
        
    def search_images(self, query_embedding, limit=SEARCH_RESULTS_LIMIT):
        """搜索相关图像"""
        return self.images_collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
        )
        
    def get_paper_by_id(self, paper_id):
        """通过ID获取文献"""
        return self.papers_collection.get(ids=[paper_id])
        
    def get_image_by_id(self, image_id):
        """通过ID获取图像"""
        return self.images_collection.get(ids=[image_id])
        
    def delete_paper(self, paper_id):
        """删除文献"""
        self.papers_collection.delete(ids=[paper_id])
        
    def delete_image(self, image_id):
        """删除图像"""
        self.images_collection.delete(ids=[image_id])