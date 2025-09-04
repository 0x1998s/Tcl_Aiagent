"""
向量检索服务
支持ChromaDB、图混合检索和语义搜索
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import networkx as nx

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from utils.logger import get_logger
from core.config import Settings

logger = get_logger(__name__)


class VectorService:
    """向量检索服务"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.chroma_client = None
        self.collections = {}
        self.embedding_model = None
        self.knowledge_graph = nx.DiGraph()
        
    async def initialize(self):
        """初始化向量服务"""
        logger.info("初始化向量检索服务...")
        
        # 初始化ChromaDB
        if CHROMA_AVAILABLE:
            await self._init_chromadb()
        else:
            logger.warning("ChromaDB不可用，使用内存向量存储")
        
        # 初始化嵌入模型
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            await self._init_embedding_model()
        else:
            logger.warning("SentenceTransformers不可用，使用简化向量化")
        
        # 初始化知识图谱
        await self._init_knowledge_graph()
        
        logger.info("向量检索服务初始化完成")
    
    async def _init_chromadb(self):
        """初始化ChromaDB"""
        try:
            self.chroma_client = chromadb.Client(
                ChromaSettings(
                    chroma_server_host=self.settings.CHROMA_HOST,
                    chroma_server_http_port=self.settings.CHROMA_PORT
                )
            )
            
            # 创建默认集合
            collection_name = self.settings.CHROMA_COLLECTION_NAME
            try:
                collection = self.chroma_client.get_collection(collection_name)
            except:
                collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": "TCL AI Agent知识库"}
                )
            
            self.collections[collection_name] = collection
            logger.info(f"ChromaDB连接成功，集合: {collection_name}")
            
        except Exception as e:
            logger.error(f"ChromaDB初始化失败: {str(e)}")
            self.chroma_client = None
    
    async def _init_embedding_model(self):
        """初始化嵌入模型"""
        try:
            model_name = self.settings.EMBEDDING_MODEL
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"嵌入模型加载成功: {model_name}")
            
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {str(e)}")
            self.embedding_model = None
    
    async def _init_knowledge_graph(self):
        """初始化知识图谱"""
        try:
            # 添加示例知识图谱节点和边
            entities = [
                {"id": "user", "type": "entity", "properties": {"name": "用户"}},
                {"id": "product", "type": "entity", "properties": {"name": "产品"}},
                {"id": "order", "type": "entity", "properties": {"name": "订单"}},
                {"id": "category", "type": "entity", "properties": {"name": "类别"}},
                {"id": "revenue", "type": "metric", "properties": {"name": "收入"}},
                {"id": "conversion_rate", "type": "metric", "properties": {"name": "转化率"}}
            ]
            
            relations = [
                {"from": "user", "to": "order", "relation": "places", "weight": 1.0},
                {"from": "order", "to": "product", "relation": "contains", "weight": 1.0},
                {"from": "product", "to": "category", "relation": "belongs_to", "weight": 0.8},
                {"from": "order", "to": "revenue", "relation": "generates", "weight": 0.9},
                {"from": "user", "to": "conversion_rate", "relation": "affects", "weight": 0.7}
            ]
            
            # 构建图
            for entity in entities:
                self.knowledge_graph.add_node(
                    entity["id"], 
                    **entity["properties"],
                    node_type=entity["type"]
                )
            
            for relation in relations:
                self.knowledge_graph.add_edge(
                    relation["from"],
                    relation["to"],
                    relation=relation["relation"],
                    weight=relation["weight"]
                )
            
            logger.info(f"知识图谱初始化完成: {len(entities)}个节点, {len(relations)}条边")
            
        except Exception as e:
            logger.error(f"知识图谱初始化失败: {str(e)}")
    
    async def embed_text(self, text: str) -> List[float]:
        """文本向量化"""
        
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(text)
                return embedding.tolist()
            except Exception as e:
                logger.error(f"文本向量化失败: {str(e)}")
        
        # 简化的向量化（基于词频）
        return await self._simple_embedding(text)
    
    async def _simple_embedding(self, text: str, dim: int = 384) -> List[float]:
        """简化的文本向量化"""
        
        # 基于字符hash的简单向量化
        import hashlib
        
        # 生成多个hash值
        hashes = []
        for i in range(dim // 4):
            hash_input = f"{text}_{i}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
            hashes.extend([
                (hash_value & 0xFF) / 255.0 - 0.5,
                ((hash_value >> 8) & 0xFF) / 255.0 - 0.5,
                ((hash_value >> 16) & 0xFF) / 255.0 - 0.5,
                ((hash_value >> 24) & 0xFF) / 255.0 - 0.5
            ])
        
        return hashes[:dim]
    
    async def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> bool:
        """添加文档到向量库"""
        
        try:
            # 生成向量
            embedding = await self.embed_text(content)
            
            if self.chroma_client:
                collection_name = collection_name or self.settings.CHROMA_COLLECTION_NAME
                collection = self.collections.get(collection_name)
                
                if collection:
                    collection.add(
                        embeddings=[embedding],
                        documents=[content],
                        metadatas=[metadata or {}],
                        ids=[doc_id]
                    )
                    logger.debug(f"文档添加成功: {doc_id}")
                    return True
            
            # 内存存储备选方案
            # 这里可以实现内存向量存储
            
            return False
            
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            return False
    
    async def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        collection_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """语义搜索"""
        
        try:
            # 生成查询向量
            query_embedding = await self.embed_text(query)
            
            if self.chroma_client:
                collection_name = collection_name or self.settings.CHROMA_COLLECTION_NAME
                collection = self.collections.get(collection_name)
                
                if collection:
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=top_k,
                        where=filters
                    )
                    
                    # 格式化结果
                    formatted_results = []
                    for i in range(len(results["ids"][0])):
                        formatted_results.append({
                            "id": results["ids"][0][i],
                            "content": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "score": 1 - results["distances"][0][i]  # 转换为相似度分数
                        })
                    
                    return formatted_results
            
            return []
            
        except Exception as e:
            logger.error(f"语义搜索失败: {str(e)}")
            return []
    
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.7,
        graph_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """混合搜索：语义搜索 + 图搜索"""
        
        try:
            # 1. 语义搜索
            semantic_results = await self.semantic_search(query, top_k * 2)
            
            # 2. 图搜索
            graph_results = await self.graph_search(query, top_k * 2)
            
            # 3. 融合结果
            combined_results = await self._combine_search_results(
                semantic_results,
                graph_results,
                semantic_weight,
                graph_weight
            )
            
            return combined_results[:top_k]
            
        except Exception as e:
            logger.error(f"混合搜索失败: {str(e)}")
            return []
    
    async def graph_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """基于知识图谱的搜索"""
        
        try:
            # 提取查询中的实体
            entities = await self._extract_entities(query)
            
            results = []
            
            for entity in entities:
                if entity in self.knowledge_graph.nodes:
                    # 获取相关节点
                    neighbors = list(self.knowledge_graph.neighbors(entity))
                    predecessors = list(self.knowledge_graph.predecessors(entity))
                    
                    related_nodes = list(set(neighbors + predecessors))
                    
                    # 计算相关性分数
                    for related_node in related_nodes:
                        edge_data = self.knowledge_graph.get_edge_data(entity, related_node) or \
                                   self.knowledge_graph.get_edge_data(related_node, entity)
                        
                        weight = edge_data.get("weight", 0.5) if edge_data else 0.5
                        
                        results.append({
                            "id": related_node,
                            "content": f"实体: {related_node}",
                            "metadata": {
                                "source_entity": entity,
                                "relation": edge_data.get("relation", "related") if edge_data else "related",
                                "node_type": self.knowledge_graph.nodes[related_node].get("node_type", "unknown")
                            },
                            "score": weight
                        })
            
            # 按分数排序
            results.sort(key=lambda x: x["score"], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"图搜索失败: {str(e)}")
            return []
    
    async def _extract_entities(self, text: str) -> List[str]:
        """从文本中提取实体"""
        
        # 简化的实体提取
        entities = []
        text_lower = text.lower()
        
        # 检查已知实体
        for node in self.knowledge_graph.nodes:
            if node.lower() in text_lower:
                entities.append(node)
        
        # 检查同义词
        synonyms = {
            "用户": ["user", "客户", "顾客"],
            "产品": ["product", "商品", "物品"],
            "订单": ["order", "购买", "交易"],
            "收入": ["revenue", "营收", "销售额"],
            "转化率": ["conversion", "转换率", "转化"]
        }
        
        for entity, syns in synonyms.items():
            for syn in syns:
                if syn in text_lower:
                    entities.append(entity)
                    break
        
        return list(set(entities))
    
    async def _combine_search_results(
        self,
        semantic_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        semantic_weight: float,
        graph_weight: float
    ) -> List[Dict[str, Any]]:
        """合并搜索结果"""
        
        # 创建结果字典
        combined = {}
        
        # 添加语义搜索结果
        for result in semantic_results:
            doc_id = result["id"]
            combined[doc_id] = {
                **result,
                "combined_score": result["score"] * semantic_weight,
                "semantic_score": result["score"],
                "graph_score": 0.0
            }
        
        # 添加图搜索结果
        for result in graph_results:
            doc_id = result["id"]
            if doc_id in combined:
                combined[doc_id]["combined_score"] += result["score"] * graph_weight
                combined[doc_id]["graph_score"] = result["score"]
            else:
                combined[doc_id] = {
                    **result,
                    "combined_score": result["score"] * graph_weight,
                    "semantic_score": 0.0,
                    "graph_score": result["score"]
                }
        
        # 转换为列表并排序
        results = list(combined.values())
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return results
    
    async def cluster_documents(
        self,
        documents: List[str],
        n_clusters: int = 5
    ) -> Dict[str, Any]:
        """文档聚类"""
        
        try:
            # 生成文档向量
            embeddings = []
            for doc in documents:
                embedding = await self.embed_text(doc)
                embeddings.append(embedding)
            
            embeddings = np.array(embeddings)
            
            # K-means聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # 组织结果
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append({
                    "index": i,
                    "document": documents[i]
                })
            
            return {
                "clusters": clusters,
                "cluster_centers": kmeans.cluster_centers_.tolist(),
                "n_clusters": n_clusters,
                "silhouette_score": await self._calculate_silhouette_score(embeddings, cluster_labels)
            }
            
        except Exception as e:
            logger.error(f"文档聚类失败: {str(e)}")
            return {"clusters": {}, "error": str(e)}
    
    async def _calculate_silhouette_score(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """计算轮廓系数"""
        try:
            from sklearn.metrics import silhouette_score
            return float(silhouette_score(embeddings, labels))
        except:
            return 0.0
    
    async def find_similar_documents(
        self,
        doc_id: str,
        top_k: int = 5,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """查找相似文档"""
        
        try:
            if self.chroma_client:
                collection_name = collection_name or self.settings.CHROMA_COLLECTION_NAME
                collection = self.collections.get(collection_name)
                
                if collection:
                    # 获取文档
                    doc_result = collection.get(ids=[doc_id])
                    if not doc_result["embeddings"]:
                        return []
                    
                    # 使用文档向量搜索相似文档
                    similar_results = collection.query(
                        query_embeddings=doc_result["embeddings"],
                        n_results=top_k + 1  # +1 因为会包含自己
                    )
                    
                    # 过滤掉自己
                    results = []
                    for i, result_id in enumerate(similar_results["ids"][0]):
                        if result_id != doc_id:
                            results.append({
                                "id": result_id,
                                "content": similar_results["documents"][0][i],
                                "metadata": similar_results["metadatas"][0][i],
                                "similarity": 1 - similar_results["distances"][0][i]
                            })
                    
                    return results[:top_k]
            
            return []
            
        except Exception as e:
            logger.error(f"查找相似文档失败: {str(e)}")
            return []
    
    async def update_knowledge_graph(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
    ) -> bool:
        """更新知识图谱"""
        
        try:
            # 添加新实体
            for entity in entities:
                self.knowledge_graph.add_node(
                    entity["id"],
                    **entity.get("properties", {}),
                    node_type=entity.get("type", "unknown")
                )
            
            # 添加新关系
            for relation in relations:
                self.knowledge_graph.add_edge(
                    relation["from"],
                    relation["to"],
                    relation=relation.get("relation", "related"),
                    weight=relation.get("weight", 0.5)
                )
            
            logger.info(f"知识图谱更新完成: +{len(entities)}个节点, +{len(relations)}条边")
            return True
            
        except Exception as e:
            logger.error(f"知识图谱更新失败: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        
        stats = {
            "embedding_model": self.embedding_model.__class__.__name__ if self.embedding_model else "Simple",
            "chroma_available": self.chroma_client is not None,
            "collections": list(self.collections.keys()),
            "knowledge_graph": {
                "nodes": len(self.knowledge_graph.nodes),
                "edges": len(self.knowledge_graph.edges)
            }
        }
        
        if self.chroma_client:
            try:
                for name, collection in self.collections.items():
                    count = collection.count()
                    stats[f"{name}_documents"] = count
            except:
                pass
        
        return stats
