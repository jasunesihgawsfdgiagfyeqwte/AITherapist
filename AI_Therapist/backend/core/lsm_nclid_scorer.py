import pandas as pd
import numpy as np
import jieba
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any, Optional
import sqlite3
import json
from datetime import datetime, timedelta


# ============ 评分系统核心类 ============

class LSMNCLIDScorer:
    """LSM & nCLiD 评分器 - 集成到LangGraph工作流"""

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.stopwords = ['的', '了', '是', '我', '你', '他', '她', '就', '不', '在', '啊', '啦', '嘛', '呢']

        # 缓存用户的对话历史向量
        self.user_vectors_cache = {}
        self.ai_vectors_cache = {}

    def get_function_words(self, text: str) -> List[str]:
        """提取功能词"""
        tokens = jieba.lcut(str(text))
        return [t for t in tokens if t in self.stopwords]

    def calc_lsm_single(self, user_text: str, ai_text: str) -> float:
        """计算单轮LSM分数"""
        words1 = self.get_function_words(user_text)
        words2 = self.get_function_words(ai_text)

        if not words1 and not words2:
            return 0.0

        overlap = len(set(words1) & set(words2))
        denom = len(set(words1) | set(words2)) + 1e-6
        return round(overlap / denom, 4)

    def calc_nclid_single(self, user_text: str, ai_text: str) -> float:
        """计算单轮nCLiD分数"""
        user_vec = self.model.encode([user_text])
        ai_vec = self.model.encode([ai_text])

        # 计算余弦距离 (1 - 余弦相似度)
        similarity = cosine_similarity(user_vec, ai_vec)[0][0]
        distance = 1 - similarity
        return round(distance, 4)

    def calc_lsm_cumulative(self, user_texts: List[str], ai_texts: List[str]) -> float:
        """计算累积LSM分数"""
        user_corpus = " ".join(user_texts)
        ai_corpus = " ".join(ai_texts)
        return self.calc_lsm_single(user_corpus, ai_corpus)

    def calc_nclid_weighted(self, user_texts: List[str], ai_texts: List[str]) -> float:
        """计算加权nCLiD分数（逐轮累积）"""
        if not user_texts or not ai_texts:
            return 0.0

        min_len = min(len(user_texts), len(ai_texts))
        if min_len == 0:
            return 0.0

        weighted_score = 0.0
        for i in range(min_len):
            current_distance = self.calc_nclid_single(user_texts[i], ai_texts[i])

            if i == 0:
                weighted_score = current_distance
            else:
                # 加权平均：当前距离与历史平均的均值
                weighted_score = (current_distance + weighted_score) / 2

        return round(weighted_score, 4)

    def score_conversation(self, user_id: str, user_input: str, ai_response: str) -> Dict[str, float]:
        """为单次对话评分，并更新累积分数"""

        # 获取用户历史对话
        conversation_history = self.get_conversation_history(user_id)

        # 计算单轮分数
        single_lsm = self.calc_lsm_single(user_input, ai_response)
        single_nclid = self.calc_nclid_single(user_input, ai_response)

        # 更新历史记录
        conversation_history["user_texts"].append(user_input)
        conversation_history["ai_texts"].append(ai_response)

        # 计算累积分数
        cumulative_lsm = self.calc_lsm_cumulative(
            conversation_history["user_texts"],
            conversation_history["ai_texts"]
        )
        cumulative_nclid = self.calc_nclid_weighted(
            conversation_history["user_texts"],
            conversation_history["ai_texts"]
        )

        # 保存更新的历史
        self.save_conversation_history(user_id, conversation_history)

        return {
            "single_lsm": single_lsm,
            "single_nclid": single_nclid,
            "cumulative_lsm": cumulative_lsm,
            "cumulative_nclid": cumulative_nclid,
            "total_rounds": len(conversation_history["user_texts"])
        }

    def get_conversation_history(self, user_id: str) -> Dict[str, List[str]]:
        """获取用户对话历史"""
        conn = sqlite3.connect('psychology_ai.db')
        cursor = conn.cursor()

        # 获取最近30天的对话（避免历史过长影响性能）
        thirty_days_ago = datetime.now() - timedelta(days=30)

        cursor.execute('''
            SELECT user_input, ai_response FROM conversations 
            WHERE user_id = ? AND created_at > ?
            ORDER BY created_at ASC
        ''', (user_id, thirty_days_ago))

        history = cursor.fetchall()
        conn.close()

        return {
            "user_texts": [h[0] for h in history],
            "ai_texts": [h[1] for h in history]
        }

    def save_conversation_history(self, user_id: str, history: Dict[str, List[str]]):
        """保存对话历史（这里实际上不需要额外保存，因为数据库已有记录）"""
        # 由于对话已经保存在conversations表中，这里只是占位符
        # 实际实现中可以用于缓存优化
        pass

    def get_user_score_summary(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """获取用户评分摘要"""
        conn = sqlite3.connect('psychology_ai.db')
        cursor = conn.cursor()

        # 获取指定天数内的评分数据
        since_date = datetime.now() - timedelta(days=days)

        cursor.execute('''
            SELECT lsm_score, nclid_score, understanding_score, created_at
            FROM conversations 
            WHERE user_id = ? AND created_at > ? AND lsm_score IS NOT NULL
            ORDER BY created_at ASC
        ''', (user_id, since_date))

        scores = cursor.fetchall()
        conn.close()

        if not scores:
            return {
                "avg_lsm": 0.0,
                "avg_nclid": 0.0,
                "avg_understanding": 0.0,
                "score_trend": "无数据",
                "total_conversations": 0
            }

        lsm_scores = [s[0] for s in scores if s[0] is not None]
        nclid_scores = [s[1] for s in scores if s[1] is not None]
        understanding_scores = [s[2] for s in scores if s[2] is not None]

        # 计算趋势（最近3次vs之前的平均）
        if len(lsm_scores) >= 6:
            recent_avg = np.mean(lsm_scores[-3:])
            previous_avg = np.mean(lsm_scores[:-3])
            trend = "上升" if recent_avg > previous_avg else "下降"
        else:
            trend = "数据不足"

        return {
            "avg_lsm": round(np.mean(lsm_scores), 4) if lsm_scores else 0.0,
            "avg_nclid": round(np.mean(nclid_scores), 4) if nclid_scores else 0.0,
            "avg_understanding": round(np.mean(understanding_scores), 2) if understanding_scores else 0.0,
            "score_trend": trend,
            "total_conversations": len(scores),
            "latest_lsm": lsm_scores[-1] if lsm_scores else 0.0,
            "latest_nclid": nclid_scores[-1] if nclid_scores else 0.0
        }


# ============ 集成到工作流状态 ============

from typing import TypedDict


class ChatStateWithScoring(TypedDict):
    """带评分的聊天状态"""

    # 基础字段
    user_id: str
    user_input: str
    character_id: str
    emotion: str
    cbt_stage: int
    ai_response: str
    understanding_score: float

    # 评分字段（新增）
    lsm_scores: Dict[str, float]  # LSM评分
    nclid_scores: Dict[str, float]  # nCLiD评分
    score_analysis: Dict[str, Any]  # 评分分析

    # 元数据
    metadata: Optional[Dict[str, Any]]


# ============ 集成到工作流节点 ============

def enhanced_memory_node(state: ChatStateWithScoring) -> ChatStateWithScoring:
    """增强的记忆节点 - 包含LSM&nCLiD评分"""

    # 初始化评分器
    scorer = LSMNCLIDScorer()

    # 计算评分
    scores = scorer.score_conversation(
        user_id=state["user_id"],
        user_input=state["user_input"],
        ai_response=state["ai_response"]
    )

    # 更新状态
    state["lsm_scores"] = {
        "single": scores["single_lsm"],
        "cumulative": scores["cumulative_lsm"]
    }
    state["nclid_scores"] = {
        "single": scores["single_nclid"],
        "cumulative": scores["cumulative_nclid"]
    }
    state["score_analysis"] = {
        "total_rounds": scores["total_rounds"],
        "score_quality": analyze_score_quality(scores),
        "improvement_suggestions": generate_improvement_suggestions(scores)
    }

    # 保存到数据库（扩展conversations表）
    save_conversation_with_scores(state)

    return state


def analyze_score_quality(scores: Dict[str, float]) -> str:
    """分析评分质量"""
    lsm = scores["cumulative_lsm"]
    nclid = scores["cumulative_nclid"]

    if lsm > 0.6 and nclid < 0.3:
        return "优秀：高语言同步性，低语义距离"
    elif lsm > 0.4 and nclid < 0.5:
        return "良好：适中的语言同步和语义匹配"
    elif lsm < 0.3 or nclid > 0.7:
        return "需改进：语言同步性或语义匹配较低"
    else:
        return "中等：有提升空间"


def generate_improvement_suggestions(scores: Dict[str, float]) -> List[str]:
    """生成改进建议"""
    suggestions = []

    if scores["cumulative_lsm"] < 0.3:
        suggestions.append("AI可以更多地模仿用户的语言风格和用词习惯")

    if scores["cumulative_nclid"] > 0.6:
        suggestions.append("AI回复与用户输入的语义关联度可以更高")

    if scores["total_rounds"] < 5:
        suggestions.append("对话轮数较少，建议继续对话以获得更准确的评分")

    return suggestions


def save_conversation_with_scores(state: ChatStateWithScoring):
    """保存带评分的对话"""
    conn = sqlite3.connect('psychology_ai.db')
    cursor = conn.cursor()

    # 扩展conversations表（如果还没有评分字段）
    try:
        cursor.execute('ALTER TABLE conversations ADD COLUMN lsm_score REAL')
        cursor.execute('ALTER TABLE conversations ADD COLUMN nclid_score REAL')
        cursor.execute('ALTER TABLE conversations ADD COLUMN cumulative_lsm REAL')
        cursor.execute('ALTER TABLE conversations ADD COLUMN cumulative_nclid REAL')
        cursor.execute('ALTER TABLE conversations ADD COLUMN score_analysis TEXT')
    except sqlite3.OperationalError:
        pass  # 字段已存在

    # 插入对话记录
    cursor.execute('''
        INSERT INTO conversations 
        (user_id, character_id, user_input, ai_response, understanding_score, 
         cbt_stage, emotion_detected, lsm_score, nclid_score, 
         cumulative_lsm, cumulative_nclid, score_analysis, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ''', (
        state["user_id"], state["character_id"], state["user_input"],
        state["ai_response"], state["understanding_score"],
        state["cbt_stage"], state["emotion"],
        state["lsm_scores"]["single"], state["nclid_scores"]["single"],
        state["lsm_scores"]["cumulative"], state["nclid_scores"]["cumulative"],
        json.dumps(state["score_analysis"], ensure_ascii=False)
    ))

    conn.commit()
    conn.close()


# ============ 用户界面集成 ============

class EnhancedLangGraphAI:
    """增强的LangGraph AI - 包含评分系统"""

    def __init__(self):
        from backend.workflow.graph import create_workflow  # 使用增强的工作流
        self.workflow = create_workflow()
        self.scorer = LSMNCLIDScorer()

    def chat_with_character(self, user_id: str, user_input: str, character_id: str = "nomi") -> dict:
        """与AI角色对话 - 包含评分信息"""

        # 构建增强的状态
        initial_state = ChatStateWithScoring(
            user_id=user_id,
            user_input=user_input,
            character_id=character_id,
            emotion="",
            cbt_stage=1,
            ai_response="",
            understanding_score=0.0,
            lsm_scores={},
            nclid_scores={},
            score_analysis={},
            metadata={}
        )

        # 执行工作流
        result_state = self.workflow.invoke(initial_state)

        # 返回增强的结果
        return {
            'character': result_state["character_id"],
            'response': result_state["ai_response"],
            'understanding_score': result_state["understanding_score"],
            'emotion_detected': result_state["emotion"],
            'cbt_stage': result_state["cbt_stage"],

            # 新增评分信息
            'lsm_single': result_state["lsm_scores"]["single"],
            'lsm_cumulative': result_state["lsm_scores"]["cumulative"],
            'nclid_single': result_state["nclid_scores"]["single"],
            'nclid_cumulative': result_state["nclid_scores"]["cumulative"],
            'score_quality': result_state["score_analysis"]["score_quality"],
            'total_rounds': result_state["score_analysis"]["total_rounds"],
            'improvement_suggestions': result_state["score_analysis"]["improvement_suggestions"]
        }

    def get_user_score_report(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """获取用户评分报告"""
        return self.scorer.get_user_score_summary(user_id, days)


# ============ 命令行界面增强 ============

def enhanced_main():
    """增强的主程序 - 显示评分信息"""
    print("🧠 心理AI助手 - 带LSM&nCLiD评分系统")
    print("=" * 60)

    # 初始化增强系统
    ai_system = EnhancedLangGraphAI()

    user_id = input("请输入用户名：").strip() or "test_user"
    print(f"欢迎，{user_id}！")

    # 显示历史评分摘要
    score_summary = ai_system.get_user_score_report(user_id)
    if score_summary["total_conversations"] > 0:
        print(f"\n📊 你的评分摘要（最近7天）：")
        print(f"   平均LSM: {score_summary['avg_lsm']:.4f}")
        print(f"   平均nCLiD: {score_summary['avg_nclid']:.4f}")
        print(f"   评分趋势: {score_summary['score_trend']}")
        print(f"   对话总数: {score_summary['total_conversations']}")

    current_character = "nomi"
    print(f"\n当前角色：{current_character}")

    while True:
        print(f"\n[{current_character}] 你：", end=" ")
        user_input = input().strip()

        if user_input.lower() in ['exit', 'quit', '退出']:
            # 显示最终评分报告
            final_report = ai_system.get_user_score_report(user_id)
            print(f"\n📈 最终评分报告：")
            print(f"   LSM (语言同步): {final_report['latest_lsm']:.4f}")
            print(f"   nCLiD (语义距离): {final_report['latest_nclid']:.4f}")
            print(f"   对话质量: {analyze_score_quality(final_report)}")
            break

        # 处理特殊命令
        if user_input == '/scores':
            report = ai_system.get_user_score_report(user_id)
            print(f"📊 评分报告：")
            print(f"   当前LSM: {report['latest_lsm']:.4f}")
            print(f"   当前nCLiD: {report['latest_nclid']:.4f}")
            print(f"   平均理解感: {report['avg_understanding']:.2f}")
            continue

        if not user_input:
            continue

        # 处理对话
        try:
            result = ai_system.chat_with_character(user_id, user_input, current_character)

            # 显示AI回复
            print(f"\n🤖 {current_character}：{result['response']}")

            # 显示详细评分信息
            print(f"\n📊 本轮评分：")
            print(f"   理解感: {result['understanding_score']:.2f}")
            print(f"   LSM: {result['lsm_single']:.4f} (累积: {result['lsm_cumulative']:.4f})")
            print(f"   nCLiD: {result['nclid_single']:.4f} (累积: {result['nclid_cumulative']:.4f})")
            print(f"   质量评价: {result['score_quality']}")
            print(f"   对话轮数: {result['total_rounds']}")

            # 显示改进建议
            if result['improvement_suggestions']:
                print(f"💡 改进建议: {'; '.join(result['improvement_suggestions'])}")

        except Exception as e:
            print(f"❌ 错误：{str(e)}")


if __name__ == "__main__":
    enhanced_main()