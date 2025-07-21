import json
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# 下载依赖（只需要跑一次）
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# 加载英文停用词
stop_words = set(stopwords.words('english'))

# 读取 JSON 数据
with open("ESConv.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 合并所有对话文本
all_text = ""
for item in data:
    for turn in item["dialog"]:
        content = turn.get("content", "").strip()
        if content:
            all_text += content + " "

# 分词
words = word_tokenize(all_text)
words = [w.lower() for w in words if w.isalpha()]  # 去除标点等非字母
word_freq = Counter(words)

# 参数
MIN_COUNT = 5
TOP_N = 200

# 提取频率高的词
common_words = [(w, c) for w, c in word_freq.items() if c >= MIN_COUNT]
common_words.sort(key=lambda x: x[1], reverse=True)
top_words = common_words[:TOP_N]

# 词性标注
tagged_words = nltk.pos_tag([w for w, _ in top_words])

# 输出词 + 词性 + 频率
final_words = []
for (word, freq), (_, pos) in zip(top_words, tagged_words):
    final_words.append({
        "word": word,
        "pos": pos,
        "freq": freq
    })
    print(f"{word:15s} {pos:6s} {freq}")

# 保存为 JSON
with open("high_freq_words_en.json", "w", encoding="utf-8") as f:
    json.dump(final_words, f, indent=2)
