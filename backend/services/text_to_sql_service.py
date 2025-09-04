"""
Text-to-SQL微调服务
支持模型微调、推理和评估
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    Trainer, TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import pandas as pd
from sklearn.metrics import accuracy_score

from utils.logger import get_logger
from core.config import Settings

logger = get_logger(__name__)


class TextToSQLService:
    """Text-to-SQL微调和推理服务"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = "microsoft/DialoGPT-medium"  # 可替换为专门的Text-to-SQL模型
        self.tokenizer = None
        self.model = None
        self.training_data = []
        self.schema_info = {}
        
    async def initialize(self):
        """初始化Text-to-SQL服务"""
        logger.info("初始化Text-to-SQL服务...")
        
        try:
            # 加载预训练模型和分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # 添加特殊token
            special_tokens = {
                "additional_special_tokens": [
                    "[TABLE]", "[COLUMN]", "[VALUE]", "[SQL]",
                    "[SELECT]", "[FROM]", "[WHERE]", "[GROUP BY]", "[ORDER BY]"
                ]
            }
            self.tokenizer.add_special_tokens(special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            # 加载数据库schema信息
            await self._load_schema_info()
            
            # 加载训练数据
            await self._load_training_data()
            
            logger.info("Text-to-SQL服务初始化完成")
            
        except Exception as e:
            logger.error(f"Text-to-SQL服务初始化失败: {str(e)}")
            raise
    
    async def _load_schema_info(self):
        """加载数据库schema信息"""
        # 这里加载数据库表结构信息
        self.schema_info = {
            "tables": {
                "users": {
                    "columns": ["id", "username", "email", "created_at"],
                    "types": ["INTEGER", "VARCHAR", "VARCHAR", "TIMESTAMP"],
                    "description": "用户表"
                },
                "orders": {
                    "columns": ["id", "user_id", "product_id", "amount", "order_date"],
                    "types": ["INTEGER", "INTEGER", "INTEGER", "DECIMAL", "DATE"],
                    "description": "订单表"
                },
                "products": {
                    "columns": ["id", "name", "category", "price"],
                    "types": ["INTEGER", "VARCHAR", "VARCHAR", "DECIMAL"],
                    "description": "产品表"
                }
            },
            "relationships": [
                {"from": "orders.user_id", "to": "users.id"},
                {"from": "orders.product_id", "to": "products.id"}
            ]
        }
    
    async def _load_training_data(self):
        """加载训练数据"""
        # 示例训练数据
        self.training_data = [
            {
                "question": "查询所有用户的数量",
                "sql": "SELECT COUNT(*) FROM users",
                "schema": self._format_schema()
            },
            {
                "question": "查询上个月的订单总金额",
                "sql": "SELECT SUM(amount) FROM orders WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH)",
                "schema": self._format_schema()
            },
            {
                "question": "查询每个产品类别的销售额",
                "sql": "SELECT p.category, SUM(o.amount) FROM products p JOIN orders o ON p.id = o.product_id GROUP BY p.category",
                "schema": self._format_schema()
            },
            {
                "question": "查询购买次数最多的用户",
                "sql": "SELECT u.username, COUNT(o.id) as order_count FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.id ORDER BY order_count DESC LIMIT 1",
                "schema": self._format_schema()
            }
        ]
    
    def _format_schema(self) -> str:
        """格式化数据库schema为文本"""
        schema_text = "[SCHEMA] "
        for table_name, table_info in self.schema_info["tables"].items():
            schema_text += f"[TABLE] {table_name} "
            for col, col_type in zip(table_info["columns"], table_info["types"]):
                schema_text += f"[COLUMN] {col} {col_type} "
        return schema_text
    
    def prepare_training_dataset(self) -> Dataset:
        """准备训练数据集"""
        
        def tokenize_function(examples):
            # 输入：问题 + schema
            inputs = [f"{ex['schema']} [QUESTION] {ex['question']}" for ex in examples]
            # 输出：SQL
            targets = [f"[SQL] {ex['sql']}" for ex in examples]
            
            model_inputs = self.tokenizer(
                inputs, 
                max_length=512, 
                truncation=True, 
                padding=True
            )
            
            labels = self.tokenizer(
                targets,
                max_length=256,
                truncation=True,
                padding=True
            )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # 转换为Dataset格式
        dataset = Dataset.from_list(self.training_data)
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    async def fine_tune_model(
        self,
        output_dir: str = "./models/text_to_sql_finetuned",
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        batch_size: int = 4
    ) -> Dict[str, Any]:
        """微调Text-to-SQL模型"""
        
        logger.info("开始Text-to-SQL模型微调...")
        
        try:
            # 准备数据集
            train_dataset = self.prepare_training_dataset()
            
            # 训练参数
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_steps=100,
                logging_steps=10,
                save_steps=500,
                evaluation_strategy="no",  # 简化版本，不使用验证集
                save_total_limit=2,
                remove_unused_columns=False,
            )
            
            # 数据整理器
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                padding=True
            )
            
            # 创建训练器
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
            )
            
            # 开始训练
            train_result = trainer.train()
            
            # 保存模型
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info("Text-to-SQL模型微调完成")
            
            return {
                "status": "success",
                "training_loss": train_result.training_loss,
                "model_path": output_dir,
                "training_samples": len(self.training_data)
            }
            
        except Exception as e:
            logger.error(f"模型微调失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def generate_sql(
        self, 
        question: str, 
        max_length: int = 256,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """根据自然语言问题生成SQL"""
        
        try:
            # 构造输入
            schema_text = self._format_schema()
            input_text = f"{schema_text} [QUESTION] {question}"
            
            # 编码输入
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # 生成SQL
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码输出
            generated_sql = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # 清理SQL
            sql = self._clean_generated_sql(generated_sql)
            
            return {
                "question": question,
                "generated_sql": sql,
                "confidence": self._calculate_confidence(question, sql),
                "schema_used": schema_text
            }
            
        except Exception as e:
            logger.error(f"SQL生成失败: {str(e)}")
            return {
                "question": question,
                "generated_sql": "",
                "error": str(e),
                "confidence": 0.0
            }
    
    def _clean_generated_sql(self, raw_sql: str) -> str:
        """清理生成的SQL"""
        # 移除特殊标记
        sql = raw_sql.replace("[SQL]", "").strip()
        
        # 基本格式化
        sql = sql.replace("  ", " ")  # 移除多余空格
        
        # 确保SQL以分号结尾
        if not sql.endswith(";"):
            sql += ";"
        
        return sql
    
    def _calculate_confidence(self, question: str, sql: str) -> float:
        """计算生成SQL的置信度"""
        # 简单的置信度计算逻辑
        confidence = 0.5  # 基础置信度
        
        # 检查SQL关键字
        sql_keywords = ["SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY"]
        for keyword in sql_keywords:
            if keyword in sql.upper():
                confidence += 0.1
        
        # 检查表名是否存在
        for table_name in self.schema_info["tables"].keys():
            if table_name in sql.lower():
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    async def evaluate_model(self, test_data: List[Dict[str, str]]) -> Dict[str, float]:
        """评估模型性能"""
        
        correct_predictions = 0
        total_predictions = len(test_data)
        
        for example in test_data:
            result = await self.generate_sql(example["question"])
            generated_sql = result["generated_sql"]
            expected_sql = example["sql"]
            
            # 简单的SQL匹配（实际应用中需要更复杂的评估）
            if self._normalize_sql(generated_sql) == self._normalize_sql(expected_sql):
                correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions
        }
    
    def _normalize_sql(self, sql: str) -> str:
        """标准化SQL用于比较"""
        return sql.upper().strip().replace("  ", " ").rstrip(";")
    
    async def add_training_example(
        self, 
        question: str, 
        sql: str, 
        feedback: str = "positive"
    ):
        """添加训练样本（用于在线学习）"""
        
        new_example = {
            "question": question,
            "sql": sql,
            "schema": self._format_schema(),
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }
        
        self.training_data.append(new_example)
        
        # 可以选择持久化到文件
        await self._save_training_data()
        
        logger.info(f"添加训练样本: {question[:50]}...")
    
    async def _save_training_data(self):
        """保存训练数据到文件"""
        try:
            os.makedirs("./data/text_to_sql", exist_ok=True)
            with open("./data/text_to_sql/training_data.json", "w", encoding="utf-8") as f:
                json.dump(self.training_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存训练数据失败: {str(e)}")
    
    async def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "training_samples": len(self.training_data),
            "schema_tables": list(self.schema_info["tables"].keys()),
            "model_size": self.model.num_parameters() if self.model else 0,
            "tokenizer_vocab_size": len(self.tokenizer) if self.tokenizer else 0
        }
