import os
import sys
import time
from typing import Dict

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 尝试导入评分系统
try:
    from backend.core.lsm_nclid_scorer import LSMNCLIDScorer
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("💡 请确保所有模块文件都存在并且语法正确")
    sys.exit(1)


def initialize_database():
    """初始化 SQLite 数据库（创建 conversations 表）"""
    import sqlite3
    conn = sqlite3.connect('psychology_ai.db')
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        character_id TEXT,
        user_input TEXT,
        ai_response TEXT,
        understanding_score REAL,
        cbt_stage INTEGER,
        emotion_detected TEXT,
        lsm_score REAL,
        nclid_score REAL,
        cumulative_lsm REAL,
        cumulative_nclid REAL,
        score_analysis TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    conn.commit()
    conn.close()
    print("📂 数据库结构已初始化")


class BasicAISystem:
    """基础AI系统 - 最小化可运行版本"""

    def __init__(self):
        print("🔄 初始化AI系统...")

        try:
            # 初始化数据库
            initialize_database()

            # 初始化评分器
            self.scorer = LSMNCLIDScorer()
            print("✅ 评分系统初始化成功")
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            raise

    def simple_chat(self, user_id: str, user_input: str) -> Dict:
        """简单对话功能 - 测试用"""

        # 模拟AI回复（先不调用真实API）
        ai_response = f"我理解你说的'{user_input}'，这是一个测试回复。"

        try:
            # 计算评分
            scores = self.scorer.score_conversation(user_id, user_input, ai_response)

            return {
                'user_input': user_input,
                'ai_response': ai_response,
                'lsm_single': scores['single_lsm'],
                'nclid_single': scores['single_nclid'],
                'lsm_cumulative': scores['cumulative_lsm'],
                'nclid_cumulative': scores['cumulative_nclid'],
                'total_rounds': scores['total_rounds']
            }
        except Exception as e:
            print(f"⚠️ 评分计算失败: {e}")
            return {
                'user_input': user_input,
                'ai_response': ai_response,
                'error': str(e)
            }
