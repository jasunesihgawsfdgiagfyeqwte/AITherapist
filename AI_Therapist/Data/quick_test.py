# quick_test.py - 验证你的数据格式
import json

# 加载你的数据文件
with open('ESConv.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"数据总量: {len(data)} 条对话")

# 检查数据结构
sample = data[0]
print("数据结构示例:")
print(f"- 情绪类型: {sample.get('emotion_type')}")
print(f"- 问题类型: {sample.get('problem_type')}")
print(f"- 对话轮数: {len(sample.get('dialog', []))}")

# 检查对话内容
dialog = sample.get('dialog', [])
for i, turn in enumerate(dialog[:3]):
    print(f"第{i+1}轮 [{turn['speaker']}]: {turn['content'][:50]}...")