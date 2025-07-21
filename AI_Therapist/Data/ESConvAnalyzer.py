#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import jieba
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import sqlite3
from datetime import datetime
import re
import os
from sentence_transformers import SentenceTransformer, util

class ESConvAnalyzer:
    """ESConv数据深度分析器"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = self.load_data()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # 初始化中文功能词库
        self.function_words = {
            "pronouns": [
                "i", "you", "me", "my", "your", "we", "they", "he", "she", "him", "her",
                "ours", "yourself", "myself", "their", "our", "his", "them", "its"
            ],
            "negations": [
                "not", "no", "never", "don", "didn", "won", "can’t", "couldn", "wasn", "don’t"
            ],
            "emotion_markers": [
                "feel", "feeling", "hope", "happy", "sorry", "understand", "love", "hate",
                "care", "worried", "hard", "sad", "angry", "upset", "frustrated", "nervous",
                "depressed", "anxious", "okay", "support", "great", "good", "bad", "better"
            ],
            "emotion_intensifiers": [
                "very", "really", "so", "too", "just", "definitely", "much", "more", "quite",
                "also", "even", "still", "only", "totally", "completely", "absolutely"
            ],
            "temporal_markers": [
                "now", "today", "always", "when", "before", "after", "soon", "sometimes",
                "still", "again", "long", "never", "year", "day", "years"
            ],
            "modal_particles": [
                "maybe", "well", "oh", "sure", "ok", "yeah", "thank", "thanks", "hmm",
                "um", "like", "hello", "hi"
            ]
        }

        # 合并所有功能词
        self.all_function_words = set()
        for words in self.function_words.values():
            self.all_function_words.update(words)

    def load_data(self) -> List[Dict]:
        """加载ESConv数据"""
        print(f"📂 加载数据: {self.data_path}")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ 成功加载 {len(data)} 条对话")
        return data

    def analyze_data_distribution(self) -> Dict[str, Any]:
        """分析数据分布"""
        print("📊 分析数据分布...")

        # 情绪类型分布
        emotion_types = [conv.get('emotion_type', 'unknown') for conv in self.data]
        emotion_dist = Counter(emotion_types)

        # 问题类型分布
        problem_types = [conv.get('problem_type', 'unknown') for conv in self.data]
        problem_dist = Counter(problem_types)

        # 对话长度分布
        dialog_lengths = [len(conv.get('dialog', [])) for conv in self.data]

        # 支持策略分析
        strategies = []
        feedback_scores = []

        for conv in self.data:
            dialog = conv.get('dialog', [])
            for turn in dialog:
                if turn.get('speaker') == 'supporter':
                    strategy = turn.get('annotation', {}).get('strategy')
                    if strategy:
                        strategies.append(strategy)

                # 收集反馈分数
                if 'feedback' in turn.get('annotation', {}):
                    try:
                        score = int(turn['annotation']['feedback'])
                        feedback_scores.append(score)
                    except:
                        pass

        strategy_dist = Counter(strategies)

        # 情绪改善分析
        emotion_improvements = []
        for conv in self.data:
            survey = conv.get('survey_score', {}).get('seeker', {})
            if 'initial_emotion_intensity' in survey and 'final_emotion_intensity' in survey:
                try:
                    initial = int(survey['initial_emotion_intensity'])
                    final = int(survey['final_emotion_intensity'])
                    improvement = initial - final
                    emotion_improvements.append(improvement)
                except:
                    pass

        analysis = {
            'total_conversations': len(self.data),
            'emotion_distribution': dict(emotion_dist),
            'problem_distribution': dict(problem_dist),
            'dialog_length_stats': {
                'mean': np.mean(dialog_lengths),
                'std': np.std(dialog_lengths),
                'min': np.min(dialog_lengths),
                'max': np.max(dialog_lengths),
                'median': np.median(dialog_lengths)
            },
            'strategy_distribution': dict(strategy_dist.most_common(10)),
            'feedback_stats': {
                'mean': np.mean(feedback_scores) if feedback_scores else 0,
                'count': len(feedback_scores),
                'distribution': dict(Counter(feedback_scores))
            },
            'emotion_improvement_stats': {
                'mean': np.mean(emotion_improvements) if emotion_improvements else 0,
                'positive_count': len([x for x in emotion_improvements if x > 0]),
                'total_count': len(emotion_improvements)
            }
        }

        return analysis

    def calculate_conversation_quality_score(self, conv: Dict) -> float:
        """计算对话质量分数"""
        dialog = conv.get('dialog', [])

        # 基础权重
        weights = {
            'length': 0.2,  # 对话长度
            'balance': 0.2,  # 对话平衡性
            'strategies': 0.25,  # 策略多样性
            'feedback': 0.25,  # 用户反馈
            'improvement': 0.1  # 情绪改善
        }

        # 1. 对话长度评分 (6-20轮为最佳)
        length = len(dialog)
        if 6 <= length <= 20:
            length_score = 1.0
        elif length < 6:
            length_score = length / 6
        else:
            length_score = max(0.5, 20 / length)

        # 2. 对话平衡性评分
        seeker_count = len([t for t in dialog if t.get('speaker') == 'seeker'])
        supporter_count = len([t for t in dialog if t.get('speaker') == 'supporter'])
        if max(seeker_count, supporter_count) > 0:
            balance_score = min(seeker_count, supporter_count) / max(seeker_count, supporter_count)
        else:
            balance_score = 0

        # 3. 策略多样性评分
        strategies = set()
        for turn in dialog:
            if turn.get('speaker') == 'supporter':
                strategy = turn.get('annotation', {}).get('strategy')
                if strategy:
                    strategies.add(strategy)

        strategy_score = min(len(strategies) / 5, 1.0)  # 5种策略为满分

        # 4. 用户反馈评分
        feedback_scores = []
        for turn in dialog:
            if 'feedback' in turn.get('annotation', {}):
                try:
                    score = int(turn['annotation']['feedback'])
                    feedback_scores.append(score)
                except:
                    pass

        if feedback_scores:
            feedback_score = np.mean(feedback_scores) / 5  # 标准化到0-1
        else:
            feedback_score = 0.5  # 默认中等分数

        # 5. 情绪改善评分
        survey = conv.get('survey_score', {}).get('seeker', {})
        improvement_score = 0.5  # 默认分数

        if 'initial_emotion_intensity' in survey and 'final_emotion_intensity' in survey:
            try:
                initial = int(survey['initial_emotion_intensity'])
                final = int(survey['final_emotion_intensity'])
                improvement = initial - final
                improvement_score = min(max(improvement / 4, 0), 1)  # 4分改善为满分
            except:
                pass

        # 计算加权总分
        total_score = (
                weights['length'] * length_score +
                weights['balance'] * balance_score +
                weights['strategies'] * strategy_score +
                weights['feedback'] * feedback_score +
                weights['improvement'] * improvement_score
        )

        return round(total_score, 3)

    def calculate_nclid(self, text1: str, text2: str) -> float:
        """计算两个文本的nCLiD语义距离（1 - 余弦相似度）"""
        emb1 = self.embedding_model.encode(text1, convert_to_tensor=True)
        emb2 = self.embedding_model.encode(text2, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(emb1, emb2).item()
        return round(1 - similarity, 4)

    def select_high_quality_conversations(self, target_count: int = 1000) -> List[Dict]:
        """选择高质量对话"""
        print(f"🔍 筛选高质量对话，目标数量: {target_count}")

        # 为每个对话计算质量分数
        scored_conversations = []

        for i, conv in enumerate(self.data):
            if i % 100 == 0:
                print(f"进度: {i}/{len(self.data)}")

            try:
                quality_score = self.calculate_conversation_quality_score(conv)
                scored_conversations.append((conv, quality_score))
            except Exception as e:
                print(f"⚠️ 处理对话 {i} 时出错: {e}")
                continue

        # 按质量分数排序
        scored_conversations.sort(key=lambda x: x[1], reverse=True)

        # 选择前N个
        selected = [item[0] for item in scored_conversations[:target_count]]

        print(f"✅ 成功筛选 {len(selected)} 条高质量对话")
        print(f"质量分数范围: {scored_conversations[0][1]:.3f} - {scored_conversations[target_count - 1][1]:.3f}")

        return selected

    def extract_function_words(self, text: str) -> List[str]:
        """提取功能词"""
        # 清理文本
        text = re.sub(r'[^\u4e00-\u9fff\w\s]', '', text.lower())

        # 分词
        tokens = jieba.lcut(text)

        # 提取功能词
        function_words = [token for token in tokens if token in self.all_function_words]

        return function_words

    def calculate_lsm_score(self, seeker_text: str, supporter_text: str) -> float:
        """计算LSM分数"""
        seeker_words = set(self.extract_function_words(seeker_text))
        supporter_words = set(self.extract_function_words(supporter_text))

        if not seeker_words and not supporter_words:
            return 0.0

        overlap = len(seeker_words & supporter_words)
        total = len(seeker_words | supporter_words)

        return overlap / total if total > 0 else 0.0

    def process_conversations_basic(self, conversations: List[Dict]) -> List[Dict]:
        """基础处理对话（不使用语义模型）"""
        print(f"⚙️ 基础处理 {len(conversations)} 条对话...")

        processed = []

        for i, conv in enumerate(conversations):
            if i % 50 == 0:
                print(f"处理进度: {i}/{len(conversations)}")

            try:
                dialog = conv.get('dialog', [])

                # 提取对话对并计算LSM
                pairs = []
                lsm_scores = []
                # 提取Nclid分数
                nclid_scores = []

                for j in range(len(dialog) - 1):
                    current = dialog[j]
                    next_turn = dialog[j + 1]

                    if current.get('speaker') == 'seeker' and next_turn.get('speaker') == 'supporter':
                        seeker_text = current.get('content', '')
                        supporter_text = next_turn.get('content', '')

                        if seeker_text.strip() and supporter_text.strip():
                            lsm = self.calculate_lsm_score(seeker_text, supporter_text)
                            nclid = self.calculate_nclid(seeker_text, supporter_text)

                            pairs.append((seeker_text, supporter_text))
                            lsm_scores.append(lsm)
                            nclid_scores.append(nclid)

                # 计算统计量
                processed_conv = {
                    'id': i,
                    'emotion_type': conv.get('emotion_type', ''),
                    'problem_type': conv.get('problem_type', ''),
                    'situation': conv.get('situation', ''),
                    'total_turns': len(dialog),
                    'conversation_pairs': len(pairs),
                    'lsm_scores': lsm_scores,
                    'avg_lsm': np.mean(lsm_scores) if lsm_scores else 0.0,
                    'avg_nclid': np.mean(nclid_scores) if nclid_scores else 1.0,
                    'quality_score': self.calculate_conversation_quality_score(conv),
                    'original_dialog': dialog,
                    'processed_at': datetime.now().isoformat()
                }

                processed.append(processed_conv)

            except Exception as e:
                print(f"⚠️ 处理对话 {i} 时出错: {e}")
                continue

        print(f"✅ 基础处理完成，共 {len(processed)} 条对话")
        return processed

    def save_results(self, processed_conversations: List[Dict],
                     analysis: Dict[str, Any], output_dir: str = "output"):
        def convert_to_builtin(obj):
            if isinstance(obj, dict):
                return {k: convert_to_builtin(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_builtin(v) for v in obj]
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        """保存处理结果"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"💾 保存结果到 {output_dir} 目录...")

        # 1. 保存处理后的对话数据
        conversations_file = os.path.join(output_dir, 'processed_conversations.json')
        with open(conversations_file, 'w', encoding='utf-8') as f:
            json.dump(processed_conversations, f, ensure_ascii=False, indent=2)

        # 2. 保存分析报告
        analysis_file = os.path.join(output_dir, 'data_analysis_report.json')
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(convert_to_builtin(analysis), f, ensure_ascii=False, indent=2)

        # 3. 保存SQLite数据库
        db_file = os.path.join(output_dir, 'esconv_processed.db')
        self.save_to_database(processed_conversations, db_file)

        # 4. 生成简要统计报告
        summary_file = os.path.join(output_dir, 'summary_report.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("ESConv数据处理摘要报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"原始数据量: {analysis['total_conversations']} 条对话\n")
            f.write(f"处理后数据量: {len(processed_conversations)} 条对话\n")
            f.write(f"平均对话长度: {analysis['dialog_length_stats']['mean']:.1f} 轮\n")
            f.write(f"平均LSM分数: {np.mean([conv['avg_lsm'] for conv in processed_conversations]):.4f}\n")
            f.write(f"平均质量分数: {np.mean([conv['quality_score'] for conv in processed_conversations]):.4f}\n\n")

            f.write("情绪类型分布:\n")
            for emotion, count in analysis['emotion_distribution'].items():
                f.write(f"  {emotion}: {count}\n")

            f.write("\n支持策略分布 (Top 5):\n")
            for strategy, count in list(analysis['strategy_distribution'].items())[:5]:
                f.write(f"  {strategy}: {count}\n")

        print(f"✅ 结果保存完成:")
        print(f"  - {conversations_file}")
        print(f"  - {analysis_file}")
        print(f"  - {db_file}")
        print(f"  - {summary_file}")

    def save_to_database(self, conversations: List[Dict], db_path: str):
        """保存到SQLite数据库"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 创建表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY,
                emotion_type TEXT,
                problem_type TEXT,
                situation TEXT,
                total_turns INTEGER,
                conversation_pairs INTEGER,
                avg_lsm REAL,
                avg_nclid REAL,
                quality_score REAL,
                dialog_json TEXT,
                processed_at TEXT
            )
        ''')

        # 插入数据
        for conv in conversations:
            cursor.execute('''
                INSERT INTO conversations 
                (emotion_type, problem_type, situation, total_turns, 
                 conversation_pairs, avg_lsm, avg_nclid, quality_score, dialog_json, processed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                conv['emotion_type'],
                conv['problem_type'],
                conv['situation'],
                conv['total_turns'],
                conv['conversation_pairs'],
                conv['avg_lsm'],
                conv['avg_nclid'],  # 👈 加入这一项
                conv['quality_score'],
                json.dumps(conv['original_dialog'], ensure_ascii=False),
                conv['processed_at']
            ))

        conn.commit()
        conn.close()


def main():
    """主处理函数"""
    print("🚀 ESConv数据分析与处理Pipeline启动")
    print("=" * 60)

    # 数据路径 - 请根据你的实际路径调整
    data_path = "ESConv.json"  # 修改为你的数据文件路径

    # 初始化分析器
    analyzer = ESConvAnalyzer(data_path)

    # 1. 分析数据分布
    print("\n📊 步骤1: 分析数据分布")
    analysis = analyzer.analyze_data_distribution()

    # 打印关键统计信息
    print(f"总对话数: {analysis['total_conversations']}")
    print(f"情绪类型: {list(analysis['emotion_distribution'].keys())}")
    print(f"平均对话长度: {analysis['dialog_length_stats']['mean']:.1f} 轮")
    print(f"用户反馈平均分: {analysis['feedback_stats']['mean']:.2f}")

    # 2. 筛选高质量对话
    print("\n🔍 步骤2: 筛选高质量对话")
    high_quality_conversations = analyzer.select_high_quality_conversations(target_count=1000)

    # 3. 基础处理（LSM计算）
    print("\n⚙️ 步骤3: 基础处理和LSM计算")
    processed_conversations = analyzer.process_conversations_basic(high_quality_conversations)

    # 4. 保存结果
    print("\n💾 步骤4: 保存处理结果")
    analyzer.save_results(processed_conversations, analysis)

    # 5. 打印最终总结
    print("\n🎉 处理完成! 总结:")
    print("=" * 60)
    print(f"✅ 原始数据: {len(analyzer.data)} 条对话")
    print(f"✅ 筛选后: {len(high_quality_conversations)} 条高质量对话")
    print(f"✅ 处理完成: {len(processed_conversations)} 条对话")
    print(f"✅ 平均LSM分数: {np.mean([conv['avg_lsm'] for conv in processed_conversations]):.4f}")
    print(f"✅ 平均质量分数: {np.mean([conv['quality_score'] for conv in processed_conversations]):.4f}")

    print("\n📁 生成的文件:")
    print("  - output/processed_conversations.json")
    print("  - output/data_analysis_report.json")
    print("  - output/esconv_processed.db")
    print("  - output/summary_report.txt")


if __name__ == "__main__":
    main()