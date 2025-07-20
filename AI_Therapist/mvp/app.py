# 最小化可行产品(MVP) - 从0开始的起点
# 基于你现有的 app.py，添加最核心的新功能

import os
import json
import time
import random
from dotenv import load_dotenv
from openai import OpenAI
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional

# ============ 配置和初始化 ============

load_dotenv("../API.env")
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")
MOONSHOT_BASE_URL = os.getenv("MOONSHOT_BASE_URL")

client = OpenAI(api_key=MOONSHOT_API_KEY, base_url=MOONSHOT_BASE_URL)


# ============ 数据库初始化 ============

def init_database():
    """初始化SQLite数据库"""
    conn = sqlite3.connect('psychology_ai.db')
    cursor = conn.cursor()

    # 用户表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 对话记录表（升级版chat_logs）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            character_id TEXT DEFAULT 'nomi',
            user_input TEXT,
            ai_response TEXT,
            understanding_score REAL,
            cbt_stage INTEGER DEFAULT 1,
            emotion_detected TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 用户心理画像表（简化版）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            current_cbt_stage INTEGER DEFAULT 1,
            total_conversations INTEGER DEFAULT 0,
            avg_understanding_score REAL DEFAULT 0,
            dominant_emotions TEXT,
            last_assessment_date TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # AI角色配置表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ai_characters (
            character_id TEXT PRIMARY KEY,
            character_name TEXT,
            system_prompt TEXT,
            greeting_message TEXT,
            is_active BOOLEAN DEFAULT TRUE
        )
    ''')

    conn.commit()
    conn.close()


def insert_default_characters():
    """插入默认AI角色"""
    conn = sqlite3.connect('psychology_ai.db')
    cursor = conn.cursor()

    characters = [
        {
            'id': 'nomi',
            'name': 'Nomi',
            'prompt': '''你是Nomi，一个温暖理解的AI朋友。
特点：高度共情、善于倾听、温柔支持
风格：温暖而有深度，像贴心的姐姐
请严格按照JSON格式回复：{"text": "你的回复内容"}''',
            'greeting': '嗨，我是Nomi~ 有什么想聊的吗？'
        },
        {
            'id': 'mira',
            'name': 'Mira',
            'prompt': '''你是Mira，一个充满活力的AI伙伴。
特点：积极乐观、充满能量、行动导向
风格：活泼有趣但不失温暖，像运动教练
请严格按照JSON格式回复：{"text": "你的回复内容"}''',
            'greeting': '嗨！我是Mira！准备好迎接美好的一天了吗？'
        }
    ]

    for char in characters:
        cursor.execute('''
            INSERT OR REPLACE INTO ai_characters 
            (character_id, character_name, system_prompt, greeting_message)
            VALUES (?, ?, ?, ?)
        ''', (char['id'], char['name'], char['prompt'], char['greeting']))

    conn.commit()
    conn.close()


# ============ 核心功能类 ============

class MinimalPsychologyAI:
    def __init__(self):
        self.cbt_stages = {
            1: "情绪识别：你现在感受到的情绪是什么？",
            2: "自动思维识别：你脑中闪现了什么想法？",
            3: "思维挑战：这种想法有多少证据支持？",
            4: "替代性思维：你是否可以换个角度思考？",
            5: "日常练习：下次遇到类似情况，你可以怎样应对？"
        }

    def get_user_profile(self, user_id: str) -> Dict:
        """获取用户画像"""
        conn = sqlite3.connect('psychology_ai.db')
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM user_profiles WHERE user_id = ?', (user_id,))
        profile = cursor.fetchone()

        if not profile:
            # 创建新用户画像
            cursor.execute('''
                INSERT INTO user_profiles (user_id) VALUES (?)
            ''', (user_id,))
            conn.commit()
            profile = (user_id, 1, 0, 0, '', None, datetime.now())

        conn.close()

        return {
            'user_id': profile[0],
            'current_cbt_stage': profile[1],
            'total_conversations': profile[2],
            'avg_understanding_score': profile[3],
            'dominant_emotions': profile[4],
            'last_assessment_date': profile[5]
        }

    def analyze_emotion(self, user_input: str) -> str:
        """改进的情绪分析 - 处理否定词"""

        # 否定词检测
        negation_words = ['不', '没', '别', '无', '不太', '不是', '不够']
        has_negation = any(neg_word in user_input for neg_word in negation_words)

        # 特殊短语优先检测（更准确）
        if '不开心' in user_input or '不太开心' in user_input:
            return '抑郁'
        if '不高兴' in user_input:
            return '抑郁'
        if '心情不好' in user_input:
            return '抑郁'
        if '提不起精神' in user_input:
            return '抑郁'

        # 原有的关键词检测
        emotion_keywords = {
            '抑郁': ['抑郁', '沮丧', '难过', '悲伤', '绝望', '无助', '低落', '郁闷'],
            '焦虑': ['焦虑', '担心', '紧张', '不安', '恐惧', '害怕'],
            '愤怒': ['愤怒', '生气', '恼火', '激怒', '暴躁'],
            '内疚': ['内疚', '后悔', '羞愧', '自责', '愧疚'],
            '快乐': ['快乐', '高兴', '开心', '愉快', '兴奋']
        }

        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in user_input:
                    # 如果有否定词且是积极情绪，转为负面情绪
                    if has_negation and emotion == '快乐':
                        return '抑郁'
                    elif not has_negation:
                        return emotion

        return '中性'

    def should_advance_cbt_stage(self, user_input: str, current_stage: int) -> bool:
        """判断是否应该进入下一个CBT阶段"""
        stage_indicators = {
            1: ['感到', '情绪', '心情'],  # 情绪识别
            2: ['想法', '觉得', '认为'],  # 自动思维
            3: ['也许', '可能', '不一定'],  # 思维挑战
            4: ['换个角度', '另一种', '其实'],  # 替代思维
            5: ['下次', '以后', '计划']  # 日常练习
        }

        indicators = stage_indicators.get(current_stage, [])
        return any(indicator in user_input for indicator in indicators)

    def generate_cbt_guidance(self, current_stage: int) -> str:
        """生成CBT指导"""
        return f"\n\n💡 CBT提示：{self.cbt_stages[current_stage]}"

    def chat_with_character(self, user_id: str, user_input: str, character_id: str = 'nomi') -> Dict:
        """与AI角色对话"""

        # 获取用户画像
        profile = self.get_user_profile(user_id)

        # 分析情绪
        detected_emotion = self.analyze_emotion(user_input)

        # 获取角色配置
        conn = sqlite3.connect('psychology_ai.db')
        cursor = conn.cursor()
        cursor.execute('SELECT system_prompt FROM ai_characters WHERE character_id = ?', (character_id,))
        character_prompt = cursor.fetchone()[0]

        # 构建对话历史
        cursor.execute('''
            SELECT user_input, ai_response FROM conversations 
            WHERE user_id = ? AND character_id = ?
            ORDER BY created_at DESC LIMIT 3
        ''', (user_id, character_id))
        recent_history = cursor.fetchall()

        # 构建messages
        messages = [{"role": "system", "content": character_prompt}]

        # 添加最近对话历史
        for hist in reversed(recent_history):
            messages.append({"role": "user", "content": hist[0]})
            messages.append({"role": "assistant", "content": hist[1]})

        # 添加CBT上下文
        current_stage = profile['current_cbt_stage']
        if current_stage <= 5:
            cbt_context = f"当前CBT阶段：{self.cbt_stages[current_stage]}"
            messages.append({"role": "system", "content": cbt_context})

        # 当前用户输入
        messages.append({"role": "user", "content": user_input})

        # 调用API生成回复
        start_time = time.time()
        response = client.chat.completions.create(
            model="moonshot-v1-128k",
            messages=messages,
            temperature=0.8,
            max_tokens=200
        )
        elapsed_time = time.time() - start_time

        # 解析回复
        try:
            content = json.loads(response.choices[0].message.content)
            ai_response = content.get("text", "").strip()
        except json.JSONDecodeError:
            ai_response = response.choices[0].message.content.strip()

        # 计算理解感评分（简化版）
        understanding_score = min(5.0, len(ai_response) / 20 + random.uniform(3.0, 4.5))

        # 判断是否需要CBT指导
        needs_cbt = detected_emotion in ['焦虑', '抑郁', '愤怒', '内疚']
        if needs_cbt and current_stage <= 5:
            cbt_guidance = self.generate_cbt_guidance(current_stage)
            ai_response += cbt_guidance

            # 检查是否可以进入下一阶段
            if self.should_advance_cbt_stage(user_input, current_stage):
                new_stage = min(current_stage + 1, 5)
                cursor.execute('''
                    UPDATE user_profiles 
                    SET current_cbt_stage = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (new_stage, user_id))

        # 保存对话记录
        cursor.execute('''
            INSERT INTO conversations 
            (user_id, character_id, user_input, ai_response, understanding_score, 
             cbt_stage, emotion_detected, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (user_id, character_id, user_input, ai_response,
              understanding_score, current_stage, detected_emotion))

        # 更新用户画像统计
        cursor.execute('''
            UPDATE user_profiles 
            SET total_conversations = total_conversations + 1,
                avg_understanding_score = (
                    SELECT AVG(understanding_score) FROM conversations WHERE user_id = ?
                ),
                dominant_emotions = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ?
        ''', (user_id, detected_emotion, user_id))

        conn.commit()
        conn.close()

        return {
            'character': character_id,
            'response': ai_response,
            'understanding_score': understanding_score,
            'emotion_detected': detected_emotion,
            'cbt_stage': current_stage,
            'elapsed_time': elapsed_time
        }

    def get_user_summary(self, user_id: str) -> Dict:
        """获取用户摘要"""
        conn = sqlite3.connect('psychology_ai.db')
        cursor = conn.cursor()

        # 获取基础统计
        cursor.execute('''
            SELECT COUNT(*) as total_conversations,
                   AVG(understanding_score) as avg_score,
                   MAX(created_at) as last_chat
            FROM conversations WHERE user_id = ?
        ''', (user_id,))
        stats = cursor.fetchone()

        # 获取情绪统计
        cursor.execute('''
            SELECT emotion_detected, COUNT(*) as count
            FROM conversations WHERE user_id = ? AND emotion_detected != '中性'
            GROUP BY emotion_detected
            ORDER BY count DESC
        ''', (user_id,))
        emotions = cursor.fetchall()

        # 获取用户画像
        profile = self.get_user_profile(user_id)

        conn.close()

        return {
            'total_conversations': stats[0] if stats[0] else 0,
            'avg_understanding_score': round(stats[1], 2) if stats[1] else 0,
            'last_chat': stats[2],
            'current_cbt_stage': profile['current_cbt_stage'],
            'emotion_distribution': dict(emotions),
            'cbt_progress': f"{profile['current_cbt_stage']}/5"
        }


# ============ 命令行界面 ============

def main():
    """主程序 - 命令行版本"""
    print("🧠 心理AI助手 - 最小化版本")
    print("支持功能：多AI角色、CBT指导、情绪分析、用户画像")
    print("=" * 50)

    # 初始化
    init_database()
    insert_default_characters()
    ai_system = MinimalPsychologyAI()

    # 用户登录
    user_id = input("请输入用户名：").strip() or "test_user"
    print(f"欢迎，{user_id}！")

    # 显示可用角色
    conn = sqlite3.connect('psychology_ai.db')
    cursor = conn.cursor()
    cursor.execute('SELECT character_id, character_name FROM ai_characters WHERE is_active = 1')
    characters = cursor.fetchall()
    conn.close()

    print("\n可用AI角色：")
    for char in characters:
        print(f"  {char[0]} - {char[1]}")

    current_character = 'nomi'
    print(f"\n当前角色：{current_character}")

    while True:
        print(f"\n[{current_character}] 你：", end=" ")
        user_input = input().strip()

        if user_input.lower() in ['exit', 'quit', '退出']:
            # 显示用户摘要
            summary = ai_system.get_user_summary(user_id)
            print(f"\n📊 对话摘要：")
            print(f"总对话数：{summary['total_conversations']}")
            print(f"平均理解感：{summary['avg_understanding_score']}")
            print(f"CBT进度：{summary['cbt_progress']}")
            print(f"主要情绪：{summary['emotion_distribution']}")
            break

        # 切换角色
        if user_input.startswith('/switch '):
            new_character = user_input.split(' ')[1]
            if any(char[0] == new_character for char in characters):
                current_character = new_character
                print(f"已切换到 {new_character}")
                continue
            else:
                print("角色不存在")
                continue

        # 显示帮助
        if user_input == '/help':
            print("命令：")
            print("  /switch <角色名> - 切换AI角色")
            print("  /summary - 查看对话摘要")
            print("  exit/quit - 退出程序")
            continue

        # 查看摘要
        if user_input == '/summary':
            summary = ai_system.get_user_summary(user_id)
            print(f"📊 当前状态：")
            print(f"对话数：{summary['total_conversations']}")
            print(f"CBT阶段：{summary['cbt_progress']}")
            print(f"理解感：{summary['avg_understanding_score']}")
            continue

        if not user_input:
            continue

        # 处理对话
        try:
            result = ai_system.chat_with_character(user_id, user_input, current_character)

            print(f"\n🤖 {current_character}：{result['response']}")
            print(
                f"📊 理解感：{result['understanding_score']:.2f} | 情绪：{result['emotion_detected']} | CBT阶段：{result['cbt_stage']}")

        except Exception as e:
            print(f"❌ 错误：{str(e)}")


if __name__ == "__main__":
    main()