"""文档处理模块，负责PDF解析、分类和管理"""
import os
import hashlib
from pathlib import Path
from pypdf import PdfReader
from .embedding import TextEmbedding
from .vector_db import VectorDB
from .utils import logger, is_pdf_file, create_topic_directories, move_file_to_topic, get_all_files_in_directory
from config import *
import datetime
import numpy as np

class DocumentProcessor:
    def __init__(self):
        self.text_embedder = TextEmbedding()
        self.vector_db = VectorDB()
        self.topic_short_names = ["CV", "NLP", "RL", "Multimodal"]
        create_topic_directories(PAPERS_DIR, self.topic_short_names + ["Other"])
        
    def extract_text_from_pdf(self, pdf_path, include_paragraphs=False):
        """从PDF中提取文本
        
        返回格式: {"title": str, "abstract": str, "keywords": str, "introduction": str, "full_text": str, "paragraphs": list}
        paragraphs: [{"text": str, "page": int}]
        """
        try:
            reader = PdfReader(str(pdf_path))
            full_text = ""
            page_texts = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
                    page_texts.append((i + 1, page_text))
            
            # 提取标题
            lines = full_text.split('\n')
            title = ""
            
            skip_keywords = ['copyright', '©', 'all rights reserved', 'doi', 'provided proper attribution', 
                           'google hereby grants', 'arxiv', 'permission', 'license', 'abstract', 'keywords']
            for line in lines[:20]:
                line = line.strip()
                if line and len(line) > 10 and len(line) < 200:
                    line_lower = line.lower()
                    if any(keyword in line_lower for keyword in skip_keywords):
                        continue
                    if any(char.isdigit() for char in line) and (len(line) < 20 or any(x in line_lower for x in ['v', 'version', 'apr', 'jan', 'feb', 'mar', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])):
                        continue
                    title = line[:200].strip()
                    break
            
            if not title:
                for line in lines[3:15]:
                    line = line.strip()
                    if line and len(line) > 10 and len(line) < 200:
                        line_lower = line.lower()
                        if not any(keyword in line_lower for keyword in skip_keywords):
                            title = line[:200].strip()
                            break
            
            if not title:
                title = full_text[:200].strip()
            
            # 提取摘要
            abstract = ""
            full_text_lower = full_text.lower()
            abstract_keywords = ["abstract", "摘要", "summary"]
            abstract_start = -1
            for keyword in abstract_keywords:
                pos = full_text_lower.find(keyword)
                if pos != -1:
                    abstract_start = pos
                    content_start = pos + len(keyword)
                    while content_start < len(full_text) and full_text[content_start] in [':', '\n', ' ', '\t']:
                        content_start += 1
                    break
            
            if abstract_start != -1:
                abstract_end = len(full_text)
                keywords_markers_for_end = ["keywords", "key words", "关键词"]
                for marker in keywords_markers_for_end:
                    pos = full_text_lower.find(marker.lower(), abstract_start)
                    if pos != -1 and pos < abstract_end:
                        abstract_end = pos
                if abstract_end == len(full_text):
                    for marker in ["\nintroduction", "\n1.", "\n\n"]:
                        pos = full_text_lower.find(marker.lower(), abstract_start)
                        if pos != -1 and pos < abstract_end:
                            abstract_end = pos
                abstract = full_text[content_start:abstract_end].strip()
                if len(abstract) > 800:
                    abstract = abstract[:800]
            
            # 提取关键词
            keywords = ""
            keywords_markers = ["keywords", "key words", "关键词"]
            for marker in keywords_markers:
                marker_lower = marker.lower()
                pos = full_text_lower.find(marker_lower)
                if pos != -1:
                    content_start = pos + len(marker)
                    while content_start < len(full_text) and full_text[content_start] in [':', '\n', ' ', '\t']:
                        content_start += 1
                    keywords_end = len(full_text)
                    intro_markers_for_end = ["introduction", "1. introduction", "1 introduction"]
                    for end_marker in intro_markers_for_end:
                        end_pos = full_text_lower.find(end_marker.lower(), content_start)
                        if end_pos != -1 and end_pos < keywords_end:
                            keywords_end = end_pos
                    keywords = full_text[content_start:keywords_end].strip()
                    if len(keywords) > 200:
                        keywords = keywords[:200]
                    break
            
            # 提取Introduction
            introduction = ""
            intro_markers = ["introduction", "1. introduction", "1 introduction", "1 introduction\n"]
            intro_start = -1
            for marker in intro_markers:
                marker_lower = marker.lower()
                pos = full_text_lower.find(marker_lower)
                if pos != -1:
                    intro_start = pos
                    content_start = pos + len(marker)
                    while content_start < len(full_text) and full_text[content_start] in [':', '\n', ' ', '\t']:
                        content_start += 1
                    break
            
            if intro_start != -1:
                intro_end = content_start + 1500
                actual_end = len(full_text)
                end_markers = ["\n2.", "\n\n2.", "\n3.", "\n\n3."]
                for marker in end_markers:
                    pos = full_text_lower.find(marker.lower(), intro_start)
                    if pos != -1 and pos < actual_end:
                        actual_end = pos
                intro_end = min(intro_end, actual_end)
                introduction = full_text[content_start:intro_end].strip()
            
            result = {
                "title": title,
                "abstract": abstract,
                "keywords": keywords,
                "introduction": introduction,
                "full_text": full_text
            }
            
            if include_paragraphs:
                paragraphs = []
                for page_num, page_text in page_texts:
                    raw_para_texts = [p.strip() for p in page_text.split('\n\n') if p.strip()]
                    
                    for raw_para in raw_para_texts:
                        if len(raw_para) < 50:
                            continue
                        
                        if len(raw_para) > PARAGRAPH_CHUNK_SIZE:
                            chunks = []
                            for i in range(0, len(raw_para), PARAGRAPH_CHUNK_SIZE - PARAGRAPH_OVERLAP):
                                chunk_end = min(i + PARAGRAPH_CHUNK_SIZE, len(raw_para))
                                chunk = raw_para[i:chunk_end].strip()
                                if len(chunk) > 100:
                                    chunks.append(chunk)
                            for chunk in chunks:
                                paragraphs.append({
                                    "text": chunk,
                                    "page": page_num
                                })
                        else:
                            paragraphs.append({
                                "text": raw_para,
                                "page": page_num
                            })
                
                result["paragraphs"] = paragraphs[:MAX_PARAGRAPHS_PER_PDF]
            
            return result
        except Exception as e:
            logger.error(f"提取PDF文本失败 {pdf_path}: {str(e)}")
            return {"title": "", "abstract": "", "keywords": "", "introduction": "", "full_text": "", "paragraphs": []}
    
    def get_paper_id(self, file_path):
        """生成唯一的文献ID"""
        file_path = Path(file_path)
        file_info = f"{str(file_path)}|{os.path.getmtime(str(file_path))}"
        return hashlib.md5(file_info.encode()).hexdigest()
    
    def get_paragraph_id(self, paper_id, para_index):
        """生成唯一的段落ID"""
        para_info = f"{paper_id}|paragraph_{para_index}"
        return hashlib.md5(para_info.encode()).hexdigest()
    
    def classify_paper(self, paper_text, topics=None, topic_names=None):
        """将文献分类到指定主题"""
        if topics is None:
            topics = DEFAULT_TOPICS
        if topic_names is None:
            topic_names = self.topic_short_names
    
        llm_classify_result = None
        if USE_LLM_FOR_QUERY_UNDERSTANDING:
            logger.info("启用LLM分类，开始对论文内容进行主题判断...")
            llm_classify_result = self._classify_with_llm(paper_text, topic_names)
        else:
            logger.info("未启用LLM分类，直接使用非LLM方案进行分类")
        
        if llm_classify_result:
            return llm_classify_result
        
        # 关键词匹配
        keyword_result = self._classify_with_keywords(paper_text, topic_names)
        if keyword_result and keyword_result != "Other":
            return keyword_result
        
        # 嵌入相似度
        topic_embeddings = self.text_embedder.embed(topics)
        paper_embedding = self.text_embedder.embed([paper_text])[0]
        
        paper_emb = np.array(paper_embedding)
        paper_norm = np.linalg.norm(paper_emb)
        if paper_norm > 0:
            paper_emb = paper_emb / paper_norm
        
        similarities = []
        for topic_emb in topic_embeddings:
            topic_emb = np.array(topic_emb)
            topic_norm = np.linalg.norm(topic_emb)
            if topic_norm > 0:
                topic_emb = topic_emb / topic_norm
            similarity = float(np.dot(paper_emb, topic_emb))
            similarities.append(similarity)
        
        max_idx = np.array(similarities).argmax()
        max_similarity = similarities[max_idx]
        
        if max_similarity >= SIMILARITY_THRESHOLD:
            return topic_names[max_idx]
        return "Other"
    
    def _classify_with_llm(self, paper_text, topic_names):
        """使用LLM进行分类"""
        try:
            topics_str = ", ".join(topic_names)
            prompt = f"""请根据以下论文的标题和摘要，将其分类到以下主题之一：{topics_str}

论文内容：
{paper_text[:1000]}

请只返回主题名称（CV、NLP、RL、Multimodal或Other），不要其他解释。"""
            result = self.text_embedder.generate_text(prompt, max_tokens=20)
            result = result.strip().upper()
            if result in topic_names:
                logger.info(f"LLM分类结果：{result}")
                return result
            logger.warning(f"LLM分类返回无效结果：{result}，将切换至非LLM分类方案")
            return None
        except Exception as e:
            logger.warning(f"LLM分类执行失败：{str(e)}，将切换至非LLM分类方案")
            return None
    
    def _classify_with_keywords(self, paper_text, topic_names):
        """使用关键词匹配分类"""
        paper_text_lower = paper_text.lower()
        topic_keywords = {
            "CV": ["computer vision", "image", "visual", "convolutional", "cnn", "vgg", "resnet", "googlenet"],
            "NLP": ["natural language", "nlp", "transformer", "bert", "gpt", "language model"],
            "RL": ["reinforcement learning", "rl", "policy", "reward", "agent", "ppo"],
            "Multimodal": ["multimodal", "vision-language", "clip", "blip", "llava"]
        }
        scores = {}
        for topic in topic_names:
            if topic in topic_keywords:
                score = sum(1 for kw in topic_keywords[topic] if kw in paper_text_lower)
                if score > 0:
                    scores[topic] = score
        if scores:
            max_score_topic = max(scores, key=scores.get)
            logger.info(f"关键词分类结果：{max_score_topic}（匹配分数：{scores[max_score_topic]}）")
            return max_score_topic
        return "Other"
    
    def understand_query_with_llm(self, query):
        """使用LLM理解查询意图"""
        try:
            prompt = f"""你是一个学术文献搜索引擎助手。用户输入了一个查询用于从文献库中检索文献，文献库中所有文献都为深度学习领域论文，请对查询进行细粒度扩写。

要求：
1. 严格保留原始查询的核心概念和关键词
2. 围绕核心概念添加同义词、相关术语和技术词汇
3. 只进行细粒度扩写，不要向外拓展到其他领域
4. 如果查询是具体的技术名词（如transformer、CNN、RL），重点扩展该技术的相关概念、应用场景和实现细节
5. 如果查询是问题形式（如"what is..."），转换为该主题的关键技术术语和概念的组合
6. 保持扩写结果的学术性和专业性，使用深度学习领域的标准术语

用户查询：{query}

改写后的查询："""
            rewritten = self.text_embedder.generate_text(prompt, max_tokens=50)
            rewritten = rewritten.strip().strip('"').strip("'").strip()
            if not rewritten or len(rewritten.strip()) < len(query.strip()) // 2:
                logger.error(f"LLM查询理解结果无效: '{rewritten}'")
                raise ValueError("LLM查询理解结果无效，无法进行搜索")
            logger.info(f"查询理解: '{query}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.error(f"LLM查询理解失败: {str(e)}")
            raise RuntimeError(f"查询理解失败，无法进行搜索: {str(e)}")
    
    def _rank_papers_without_llm(self, query_embedding, candidate_papers):
        """非LLM论文排序：使用余弦相似度对候选论文进行排序"""
        logger.info("开始非LLM论文排序：基于查询与文档的嵌入余弦相似度")
        ranked_papers = []
        query_emb = np.array(query_embedding)
        query_norm = np.linalg.norm(query_emb)
        
        if query_norm == 0:
            logger.warning("查询嵌入为零向量，无法进行余弦相似度排序")
            return [ {**paper, "llm_score": 0.0, "llm_reason": "非LLM模式：查询嵌入无效"} for paper in candidate_papers ]
        
        # 归一化查询嵌入
        query_emb_normalized = query_emb / query_norm
        
        for paper in candidate_papers:
            # 构建文档嵌入文本（与存储时保持一致：标题+摘要+关键词）
            doc_text_parts = [
                paper.get("title", "").strip(),
                paper.get("abstract", "").strip(),
                paper.get("keywords", "").strip()
            ]
            doc_text = " ".join([part for part in doc_text_parts if part])
            
            if not doc_text:
                doc_text = paper.get("filename", "")
            
            # 生成文档嵌入并计算余弦相似度
            doc_embedding = self.text_embedder.embed([doc_text])[0]
            doc_emb = np.array(doc_embedding)
            doc_norm = np.linalg.norm(doc_emb)
            
            similarity = 0.0
            if doc_norm > 0:
                doc_emb_normalized = doc_emb / doc_norm
                similarity = float(np.dot(query_emb_normalized, doc_emb_normalized))
            
            # 转换为与LLM评分兼容的0-10分制
            compatibility_score = (similarity + 1) * 5.0
            ranked_papers.append({
                **paper,
                "llm_score": compatibility_score,
                "llm_reason": f"非LLM模式：余弦相似度={similarity:.4f}（转换为0-10分制）"
            })
        
        # 按兼容分数降序排序
        ranked_papers.sort(key=lambda x: x['llm_score'], reverse=True)
        logger.info(f"非LLM排序完成，共处理 {len(ranked_papers)} 篇候选论文")
        return ranked_papers
    
    def _rerank_papers_with_llm(self, query, candidate_papers, batch_size=5):
        """使用LLM对候选论文进行重排序和评分"""
        scored_papers = []
        
        for i in range(0, len(candidate_papers), batch_size):
            batch = candidate_papers[i:i + batch_size]
            
            # 构建批量评估的prompt
            papers_text = ""
            for idx, paper in enumerate(batch):
                abstract_text = paper.get('abstract', 'N/A')
                if len(abstract_text) > 500:
                    abstract_text = abstract_text[:500] + "..."
                paper_info = f"""
论文{idx + 1}:
- 标题: {paper.get('title', 'N/A')}
- 摘要: {abstract_text}
- 关键词: {paper.get('keywords', 'N/A')}
- 类别: {paper.get('topic', 'Unknown')}
"""
                papers_text += paper_info
            
            prompt = f"""你是一个严格的学术文献检索专家。用户提出了一个查询，需要从以下论文中找出最相关的论文。

用户查询：{query}

候选论文：
{papers_text}

请对每篇论文与查询的相关性进行严格评分（0-10分），并给出简短理由。
评分标准（必须严格遵守）：
- 10分：论文直接回答查询，是查询主题的核心论文（例如：查询"transformer"时，只有关于transformer架构的论文才能得10分）
- 7-9分：论文与查询相关，但可能不是最直接的答案（例如：查询"transformer"时，关于BERT或GPT的论文可能得7-8分）
- 4-6分：论文与查询有一定关联，但相关性较弱（例如：查询"transformer"时，关于一般深度学习的论文可能得4-6分）
- 0-3分：论文与查询基本无关（例如：查询"transformer"时，关于计算机视觉或强化学习的论文应该得0-3分）

重要提示：
1. 如果查询是具体的技术名词（如"transformer"、"CNN"、"PPO"），只有直接研究该技术的论文才能得高分
2. 如果查询是问题形式（如"what is transformer"），只有直接定义或介绍该技术的论文才能得高分
3. 对于不相关的论文，必须给出低分（0-3分），不要因为论文质量高就给高分

请按照以下格式返回结果（每篇论文一行，必须严格按照格式）：
论文1: 分数|理由
论文2: 分数|理由
论文3: 分数|理由
...

只返回评分和理由，不要其他解释。每行必须以"论文"开头，后跟数字、冒号、分数、竖线、理由。"""
            
            try:
                response = self.text_embedder.generate_text(prompt, max_tokens=500)
                logger.info(f"LLM重排序响应（批次 {i//batch_size + 1}）: {response[:200]}...")
                
                # 解析LLM响应
                import re
                lines = response.strip().split('\n')
                
                # 先尝试按论文编号匹配
                paper_scores = {}
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 匹配格式：论文X: 分数|理由 或 论文X: 分数
                    match = re.match(r'论文\s*(\d+)\s*[:：]\s*([^|]+)(?:\s*\|\s*(.+))?', line)
                    if match:
                        paper_num = int(match.group(1))
                        score_part = match.group(2).strip()
                        reason = match.group(3).strip() if match.group(3) else ""
                        
                        # 提取数字分数
                        score_match = re.search(r'(\d+(?:\.\d+)?)', score_part)
                        if score_match:
                            score = float(score_match.group(1))
                            score = max(0, min(10, score))  # 限制在0-10
                            paper_scores[paper_num] = {"score": score, "reason": reason}
                            logger.info(f"解析成功: 论文{paper_num} = {score}分, 理由: {reason[:50]}")
                
                # 为每篇论文分配分数
                for idx, paper in enumerate(batch):
                    paper_num = idx + 1
                    if paper_num in paper_scores:
                        score_info = paper_scores[paper_num]
                        scored_papers.append({
                            "id": paper.get("id", ""),
                            "path": paper.get("path", ""),
                            "filename": paper.get("filename", ""),
                            "topic": paper.get("topic", "Other"),
                            "llm_score": score_info["score"],
                            "llm_reason": score_info["reason"],
                            "title": paper.get("title", ""),
                            "abstract": paper.get("abstract", ""),
                            "keywords": paper.get("keywords", "")
                        })
                    else:
                        # 如果解析失败，给低分（避免不相关论文被误选）
                        logger.warning(f"论文{paper_num}解析失败，给予低分2.0")
                        scored_papers.append({
                            "id": paper.get("id", ""),
                            "path": paper.get("path", ""),
                            "filename": paper.get("filename", ""),
                            "topic": paper.get("topic", "Other"),
                            "llm_score": 2.0,  # 解析失败给低分
                            "llm_reason": "LLM响应解析失败",
                            "title": paper.get("title", ""),
                            "abstract": paper.get("abstract", ""),
                            "keywords": paper.get("keywords", "")
                        })
            except Exception as e:
                logger.error(f"LLM重排序失败（批次 {i//batch_size + 1}）: {str(e)}")
                # 如果LLM调用失败，给这批论文默认分数
                for paper in batch:
                    scored_papers.append({
                        "id": paper.get("id", ""),
                        "path": paper.get("path", ""),
                        "filename": paper.get("filename", ""),
                        "topic": paper.get("topic", "Other"),
                        "llm_score": 5.0,
                        "llm_reason": "LLM评估失败",
                        "title": paper.get("title", ""),
                        "abstract": paper.get("abstract", ""),
                        "keywords": paper.get("keywords", "")
                    })
        
        # 按LLM评分排序
        scored_papers.sort(key=lambda x: x['llm_score'], reverse=True)
        return scored_papers
    
    def _search_snippets_for_paper(self, paper_path, query_embedding, top_k=3):
        """针对单篇论文，搜索与查询最匹配的段落（返回带页码的片段）"""
        try:
            # 提取论文的段落（包含页码）
            extracted = self.extract_text_from_pdf(paper_path, include_paragraphs=True)
            paragraphs = extracted.get("paragraphs", [])
            if not paragraphs:
                return []
            
            # 提取段落文本，生成嵌入
            para_texts = [para["text"] for para in paragraphs]
            para_embeddings = self.text_embedder.embed(para_texts)
            if len(para_embeddings) == 0:
                return []
            
            # 计算查询与每个段落的余弦相似度
            query_emb = np.array(query_embedding)
            query_norm = np.linalg.norm(query_emb)
            if query_norm == 0:
                return []
            query_emb_normalized = query_emb / query_norm
            
            para_similarities = []
            for idx, para_emb in enumerate(para_embeddings):
                para_emb = np.array(para_emb)
                para_norm = np.linalg.norm(para_emb)
                if para_norm == 0:
                    similarity = 0.0
                else:
                    para_emb_normalized = para_emb / para_norm
                    similarity = float(np.dot(query_emb_normalized, para_emb_normalized))
                
                para_similarities.append({
                    "paragraph": paragraphs[idx],
                    "similarity": similarity,
                    "score": (similarity + 1) * 5.0  # 转换为0-10分制，保持兼容
                })
            
            # 按相似度排序，取前top_k个片段
            para_similarities.sort(key=lambda x: x["similarity"], reverse=True)
            top_snippets = para_similarities[:top_k]
            
            # 格式化返回结果
            formatted_snippets = []
            for snippet in top_snippets:
                para = snippet["paragraph"]
                formatted_snippets.append({
                    "text": para["text"][:1000],  # 限制片段长度，避免结果过大
                    "page": para["page"],
                    "similarity": snippet["similarity"],
                    "score": snippet["score"]
                })
            
            return formatted_snippets
        except Exception as e:
            logger.error(f"搜索论文片段失败 {paper_path}: {str(e)}")
            return []
    # ==============================================================================================================
    
    def process_single_paper(self, file_path, topics=None):
        """处理单篇文献：提取文本、生成嵌入、分类并存储"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"文件不存在: {file_path}")
            return None
            
        if not is_pdf_file(file_path):
            logger.error(f"不是PDF文件: {file_path}")
            return None
        
        # 提取文本（包含段落，用于后续片段搜索）
        logger.info(f"正在处理文献: {file_path}")
        extracted = self.extract_text_from_pdf(file_path, include_paragraphs=True)
        
        title = extracted.get("title", "")
        abstract = extracted.get("abstract", "")
        keywords = extracted.get("keywords", "")
        introduction = extracted.get("introduction", "")
        
        # 用于分类的文本（仅使用摘要+关键词，不包含标题和段落）
        paper_text_parts = []
        if abstract:
            paper_text_parts.append(abstract)
        if keywords:
            paper_text_parts.append(keywords)
        paper_text = "\n".join(paper_text_parts).strip()
        
        if not paper_text:
            paper_text = file_path.name
        
        # 处理topics参数
        topic_names = None
        if topics:
            if all(t in self.topic_short_names for t in topics):
                topic_names = topics
                topics = []
                name_to_desc = {
                    "CV": DEFAULT_TOPICS[0],
                    "NLP": DEFAULT_TOPICS[1],
                    "RL": DEFAULT_TOPICS[2],
                    "Multimodal": DEFAULT_TOPICS[3]
                }
                for name in topic_names:
                    if name in name_to_desc:
                        topics.append(name_to_desc[name])
            else:
                topics = topics if isinstance(topics, list) else [topics]
                topic_names = self.topic_short_names
        else:
            topics = DEFAULT_TOPICS
            topic_names = self.topic_short_names
        
        # 分类
        topic = self.classify_paper(paper_text, topics, topic_names)
        logger.info(f"文献分类结果: {topic}")
        
        # 移动文件
        new_path = move_file_to_topic(file_path, PAPERS_DIR, topic)
        
        # 生成ID
        paper_id = self.get_paper_id(new_path)
        
        # 删除旧记录
        try:
            # 先删除当前路径的记录
            self.vector_db.delete_papers_by_path(new_path)
            # 再删除所有相同文件名的记录
            self.vector_db.delete_papers_by_filename(new_path.name)
        except Exception as e:
            logger.debug(f"检查/删除旧记录时出错: {str(e)}")
        
        # 生成文档级嵌入：使用标题+摘要+关键词
        embedding_text_parts = []
        if title:
            embedding_text_parts.append(title.strip())
        if abstract:
            embedding_text_parts.append(abstract.strip())
        if keywords:
            embedding_text_parts.append(keywords.strip())
        
        embedding_text = " ".join(embedding_text_parts).strip()
        
        # 如果没有原始内容，使用文件名作为后备
        if not embedding_text or len(embedding_text.strip()) < 10:
            embedding_text = file_path.name
        
        # 直接使用标题+摘要+关键词生成向量
        logger.info(f"使用标题+摘要+关键词生成检索向量（长度：{len(embedding_text)}字符）")
        doc_embedding = self.text_embedder.embed([embedding_text])[0]
        
        # 存储文档级信息
        doc_metadata = {
            "path": str(new_path),
            "filename": new_path.name,
            "topic": topic,
            "title": title[:500],
            "abstract": abstract[:1000],
            "keywords": keywords[:200],
            "added_at": datetime.datetime.now().isoformat(),
            "type": "document"
        }
        
        # 存储文档级信息
        try:
            self.vector_db.add_papers(
                paper_ids=[paper_id],
                embeddings=[doc_embedding],
                metadatas=[doc_metadata]
            )
            logger.info(f"文献处理完成，已保存到: {new_path}")
        except Exception as e:
            logger.error(f"存储到向量数据库失败: {str(e)}")
            raise
        
        return {
            "id": paper_id,
            "path": str(new_path),
            "topic": topic
        }
    
    def batch_process_papers(self, directory, topics=None):
        """批量处理目录中的所有PDF文献"""
        directory = Path(directory)
        if not directory.exists():
            logger.error(f"目录不存在: {directory}")
            return []
        
        pdf_files = get_all_files_in_directory(directory, is_pdf_file)
        logger.info(f"发现 {len(pdf_files)} 个PDF文件待处理")
        if not pdf_files:
            logger.warning("目录中没有找到PDF文件")
            return []
        
        results = []
        from tqdm import tqdm
        for file_path in tqdm(pdf_files, desc="处理论文", unit="篇"):
            try:
                result = self.process_single_paper(file_path, topics)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"处理文件失败 {file_path}: {str(e)}")
                continue
        
        logger.info(f"批量处理完成，共成功处理 {len(results)}/{len(pdf_files)} 个文件")
        return results
    
    def organize_papers(self, papers_dir=None, topics=None):
        """一键整理论文：扫描所有PDF，重新分类并移动到正确文件夹"""
        from tqdm import tqdm
        
        if papers_dir is None:
            papers_dir = PAPERS_DIR
        papers_dir = Path(papers_dir)
        
        if not papers_dir.exists():
            logger.error(f"论文目录不存在: {papers_dir}")
            return {"success": 0, "failed": 0, "total": 0}
        
        logger.info(f"正在扫描论文目录: {papers_dir}")
        pdf_files = get_all_files_in_directory(papers_dir, is_pdf_file)
        logger.info(f"发现 {len(pdf_files)} 个PDF文件待整理")
        
        if not pdf_files:
            logger.warning("目录中没有找到PDF文件")
            return {"success": 0, "failed": 0, "total": 0}
        
        success_count = 0
        failed_count = 0
        for file_path in tqdm(pdf_files, desc="整理论文", unit="篇"):
            try:
                result = self.process_single_paper(file_path, topics)
                if result:
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"整理文件失败 {file_path}: {str(e)}")
                failed_count += 1
                continue
        
        logger.info(f"一键整理完成，共成功整理 {success_count}/{len(pdf_files)} 篇论文")
        
        # 清理重复记录：确保每个文件名只有一个记录（保留最终路径的记录）
        logger.info("正在清理重复记录...")
        cleaned_count = self._clean_duplicate_records()
        if cleaned_count > 0:
            logger.info(f"已清理 {cleaned_count} 条重复记录")
        
        return {
            "success": success_count,
            "failed": failed_count,
            "total": len(pdf_files),
            "cleaned": cleaned_count
        }
    
    def _clean_duplicate_records(self):
        """清理重复记录：对于同一文件名，只保留最终路径下的记录"""
        try:
            # 获取所有记录
            all_data = self.vector_db.papers_collection.get(include=['metadatas'])
            
            if not all_data or 'ids' not in all_data or len(all_data['ids']) == 0:
                return 0
            
            # 按文件名分组，记录每个文件名的所有记录
            filename_to_records = {}
            for i, metadata in enumerate(all_data['metadatas']):
                if not metadata:
                    continue
                filename = metadata.get('filename', '')
                if filename:
                    if filename not in filename_to_records:
                        filename_to_records[filename] = []
                    filename_to_records[filename].append({
                        'id': all_data['ids'][i],
                        'path': metadata.get('path', ''),
                        'metadata': metadata
                    })
            
            # 对于每个文件名，如果有多个记录，只保留最终路径的记录
            ids_to_delete = []
            papers_dir_normalized = str(PAPERS_DIR.resolve()).lower()
            
            for filename, records in filename_to_records.items():
                if len(records) > 1:
                    # 找到最终路径
                    final_record = None
                    for record in records:
                        path_str = record['path']
                        try:
                            path_obj = Path(path_str)
                            # 检查路径是否在PAPERS_DIR下，并且父目录是主题文件夹
                            try:
                                path_resolved = path_obj.resolve()
                                path_str_normalized = str(path_resolved).lower()
                                # 检查路径是否在papers目录下
                                if path_str_normalized.startswith(papers_dir_normalized):
                                    # 检查父目录是否是主题文件夹
                                    if len(path_obj.parts) >= 2:
                                        parent_name = path_obj.parts[-2]  # 父目录名
                                        # 检查是否是主题文件夹（CV, NLP, RL, Multimodal, Other）
                                        if parent_name in self.topic_short_names + ['Other']:
                                            final_record = record
                                            break
                            except (OSError, ValueError):
                                # 路径解析失败，跳过
                                pass
                        except (OSError, ValueError):
                            # Path构造失败，跳过
                            pass
                    
                    # 如果没找到最终路径，保留第一个
                    if not final_record:
                        final_record = records[0]
                    
                    # 删除其他记录
                    for record in records:
                        if record['id'] != final_record['id']:
                            ids_to_delete.append(record['id'])
            
            # 批量删除重复记录
            if ids_to_delete:
                self.vector_db.papers_collection.delete(ids=ids_to_delete)
                return len(ids_to_delete)
            
            return 0
        except Exception as e:
            logger.error(f"清理重复记录失败: {str(e)}")
            return 0
    
    def search_papers(self, query, limit=SEARCH_RESULTS_LIMIT, use_query_expansion=None, snippets=False):
        """使用智能搜索：向量搜索 + 排序（支持返回匹配段落和页码）
        
        Args:
            query: 搜索查询
            limit: 返回结果数量
            use_query_expansion: 是否启用查询扩展（默认跟随总开关）
            snippets: 是否返回匹配的段落和页码（--snippets参数对应）
        
        Returns:
            包含文档信息和可选片段的搜索结果
        """
        if not query:
            logger.warning("搜索查询为空")
            return []
        
        logger.info(f"开始智能搜索: '{query}'（{'返回片段' if snippets else '不返回片段'}）")
        
        # 统一查询扩展逻辑
        if use_query_expansion is None:
            use_query_expansion = USE_LLM_FOR_QUERY_UNDERSTANDING
        
        query_for_embedding = query  # 初始化查询
        if use_query_expansion and USE_LLM_FOR_QUERY_UNDERSTANDING:
            logger.info("启用LLM查询理解，开始对查询进行细粒度扩写...")
            try:
                # 调用LLM扩写查询，用于生成更全面的嵌入
                query_for_embedding = self.understand_query_with_llm(query)
            except Exception as e:
                logger.error(f"LLM查询理解执行失败，兜底使用原始查询: {str(e)}")
                query_for_embedding = query
        else:
            logger.info("未启用LLM查询理解，直接使用原始查询进行向量搜索")
        
        # 使用向量搜索找到候选论文
        candidate_limit = max(limit * 5, 30)
        
        # 生成查询嵌入
        query_embedding = self.text_embedder.embed([query_for_embedding])[0]
        query_emb = np.array(query_embedding)
        query_norm = np.linalg.norm(query_emb)
        
        if query_norm == 0:
            logger.warning("查询嵌入为零向量")
            return []
        
        query_embedding_normalized = query_emb / query_norm
        query_embedding_for_search = query_embedding_normalized.tolist()
        
        # 向量搜索获取候选论文
        logger.info(f"向量搜索获取候选论文（最多 {candidate_limit} 篇）...")
        results = self.vector_db.search_papers(query_embedding_for_search, candidate_limit)
        
        doc_ids = results['ids'][0]
        result_metadatas = results['metadatas'][0]
        
        if not doc_ids:
            logger.warning("向量搜索未找到任何论文")
            return []
        
        logger.info(f"向量搜索找到 {len(doc_ids)} 篇候选论文")
        
        # 准备候选论文的完整信息（标题+摘要+关键词）
        candidate_papers = []
        for i, doc_id in enumerate(doc_ids):
            if i >= len(result_metadatas):
                continue
            
            metadata = result_metadatas[i]
            candidate_papers.append({
                "id": doc_id,
                "path": metadata.get("path", ""),
                "filename": metadata.get("filename", ""),
                "topic": metadata.get("topic", "Other"),
                "title": metadata.get("title", ""),
                "abstract": metadata.get("abstract", ""),
                "keywords": metadata.get("keywords", "")
            })
        
        # 总开关控制排序方案
        if USE_LLM_FOR_QUERY_UNDERSTANDING:
            # 启用LLM：使用LLM重排序
            logger.info(f"启用LLM重排序，开始对 {len(candidate_papers)} 篇候选论文进行评分...")
            ranked_papers = self._rerank_papers_with_llm(query, candidate_papers, batch_size=5)
        else:
            # 禁用LLM：使用非LLM余弦相似度排序
            ranked_papers = self._rank_papers_without_llm(query_embedding, candidate_papers)
        
        # 过滤和返回结果
        # 只返回评分 >= 5.0 的论文
        filtered_results = [paper for paper in ranked_papers if paper['llm_score'] >= 5.0]
        
        # 如果过滤后没有结果，但最高分 >= 3.0，则至少返回最高分的那一篇
        if not filtered_results and ranked_papers:
            max_score_paper = max(ranked_papers, key=lambda x: x['llm_score'])
            if max_score_paper['llm_score'] >= 3.0:
                filtered_results = [max_score_paper]
                logger.warning(f"所有论文评分都低于5.0，返回最高分论文: {max_score_paper['filename']} (分数: {max_score_paper['llm_score']:.1f})")
        
        # 如果需要返回片段，为每篇论文搜索匹配的段落和页码
        final_results = []
        for paper in filtered_results[:limit]:
            paper_result = {
                "id": paper["id"],
                "path": paper["path"],
                "filename": paper["filename"],
                "topic": paper["topic"],
                "similarity": paper["llm_score"] / 10.0,  # 转换为0-1范围，保持兼容性
                "llm_score": paper["llm_score"],
                "llm_reason": paper.get("llm_reason", "")
            }
            
            # 添加匹配片段
            if snippets:
                logger.debug(f"为论文 {paper['filename']} 搜索匹配片段...")
                paper_snippets = self._search_snippets_for_paper(
                    paper_path=paper["path"],
                    query_embedding=query_embedding,
                    top_k=SNIPPETS_TOP_K
                )
                paper_result["snippets"] = paper_snippets
            
            final_results.append(paper_result)
        
        logger.info(f"搜索完成，返回 {len(final_results)} 篇最相关论文（{'包含片段' if snippets else '不包含片段'}）")
        if final_results and snippets:
            total_snippets = sum(len(paper.get("snippets", [])) for paper in final_results)
            logger.info(f"共返回 {total_snippets} 个匹配片段")
        
        return final_results
    # ==============================================================================================================