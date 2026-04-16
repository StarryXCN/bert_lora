import re
import csv
import html
import argparse

# 1. 定义清洗函数（针对商品评论优化）
def clean_comment(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    text = text.replace('\\n', '')

    text = html.unescape(text)

    # 1. 去掉首尾空格 + 把连续空格变成一个空格
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    # 2. 去掉网址
    text = re.sub(r'https?://\S+|www.\S+', '', text, flags=re.IGNORECASE)

    # 3. 去掉手机号（支持 13800138000 和 138xxxx1234）
    text = re.sub(r'1[3-9]\d{1,2}[*xX]{4}\d{4}', '', text)  # 带*号手机号
    text = re.sub(r'1[3-9]\d{9}', '', text)                # 纯11位手机号

    # 4. 去掉微信号、微信相关（最强！）
    text = re.sub(r'[微信wxWX]{1,2}\s*[:：]?\s*[a-zA-Z0-9_-]+', '', text)
    text = re.sub(r'加我|加微信|微信号|vx|VX|WX|wx', '', text)

    # 5. 去掉邮箱
    text = re.sub(r'\S+@\S+\.\S+', '', text)

    # 步骤4：【关键】过滤掉无意义的垃圾文本
    # 规则：如果文本里的中文字符占比低于30%，直接丢弃
    chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
    total_chars = len(text.replace(" ", ""))
    if total_chars == 0 or chinese_chars / total_chars < 0.3: return ""

    # 7. 最后再清理空格
    text = text.strip()

    # 8. 过滤太短、无意义的文本
    if len(text) < 10: return ""

    return text

def main():
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="评论数据清洗脚本")
    parser.add_argument("--raw_csv", required=True, type=str, help="原始CSV文件路径")
    parser.add_argument("--cleaned_csv", required=True, type=str, help="清洗后输出CSV文件路径")

    # 解析参数
    args = parser.parse_args()

    raw_dataset = []
    with open(args.raw_csv, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            sentence = row["sentence"]
            label = row["label"]
            if len(sentence) > 100 or label is None or '' == label: continue
            raw_dataset.append({"sentence": sentence, "label": label})
    cleaned_dataset = []
    for raw_data in raw_dataset:
        cleaned_sentence = clean_comment(raw_data["sentence"])
        if cleaned_sentence == "": continue
        cleaned_dataset.append({"sentence": cleaned_sentence, "label": raw_data["label"]})
    with open(args.cleaned_csv, 'w', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["sentence", "label"])
        writer.writeheader()
        writer.writerows(cleaned_dataset)
    print(f"清洗完成，清洗后总共：{len(cleaned_dataset)} 条")


# 2. 测试效果
if __name__ == "__main__":
    main()
