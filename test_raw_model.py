import csv

import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 1. 加载模型与分词器
# num_labels=2 表示二分类：0差评 1好评
model_name = "/Users/starryx/llm_lib/models--google-bert--bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    problem_type="single_label_classification"
)

model.eval()  # 预测模式

# 2. 预测函数
def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return pred

# 3. 测试（你可以随便改这里的评论）
if __name__ == "__main__":
    with open("data/cleaned_test_dataset.csv", "r", encoding='utf-8') as file:
        test_dataset = list(csv.DictReader(file))

    total_count = len(test_dataset)
    correct = 0
    for data in test_dataset:
        res = predict_sentiment(data["sentence"])
        if data["label"] == res: correct += 1
        # print(f"评论：{comment}")
        # print(f"结果：{res} （1=好评，0=差评）\n")
    accuracy = correct / total_count
    print("\n===== 原始模型 正确率 =====")
    print(f"总样本：{total_count}")
    print(f"正确：{correct}")
    print(f"正确率：{accuracy:.2%}")