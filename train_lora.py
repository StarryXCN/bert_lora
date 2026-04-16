import csv

import torch
import pandas as pd
import argparse
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch.optim import AdamW

# ===================== 命令行参数 =====================
parser = argparse.ArgumentParser()
parser.add_argument("--raw_csv_train", required=True, type=str, help="训练集CSV路径")
parser.add_argument("--raw_csv_val", required=True, type=str, help="验证集CSV路径")
parser.add_argument("--raw_csv_test", required=True, type=str, help="测试集CSV路径")
parser.add_argument("--eval_steps", required=True, type=int, help="每隔多少步验证一次")
args = parser.parse_args()

# ===================== 配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "/Users/starryx/llm_lib/models--google-bert--bert-base-chinese"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5

# LoRA 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["query", "key", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)

# ===================== 模型 =====================
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model = get_peft_model(model, lora_config)
model = model.to(DEVICE)

# ===================== 读取3个数据集 =====================

def load_csv(path):
    with open(path, "r", encoding='utf-8') as file:
        dataset = list(csv.DictReader(file))
    return {
        "text": [item["sentence"] for item in dataset],
        "label": [int(item["label"]) for item in dataset]
    }

train_df = pd.DataFrame(load_csv(args.raw_csv_train))
val_df = pd.DataFrame(load_csv(args.raw_csv_val))
test_df = pd.DataFrame(load_csv(args.raw_csv_test))


# ===================== 编码函数 =====================
def encode(texts):
    return tokenizer(
        texts, truncation=True, padding="max_length",
        max_length=MAX_LEN, return_tensors="pt"
    )


# 训练集
train_enc = encode(train_df["text"].tolist())
train_ds = TensorDataset(train_enc.input_ids, train_enc.attention_mask, torch.tensor(train_df["label"].tolist()))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# 验证集
val_enc = encode(val_df["text"].tolist())
val_ds = TensorDataset(val_enc.input_ids, val_enc.attention_mask, torch.tensor(val_df["label"].tolist()))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# 测试集
test_enc = encode(test_df["text"].tolist())
test_ds = TensorDataset(test_enc.input_ids, test_enc.attention_mask, torch.tensor(test_df["label"].tolist()))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ===================== 优化器 =====================
optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)


# ===================== 评估函数 =====================
def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    total_val_loss = 0.0
    with torch.no_grad():
        for ids, mask, label in loader:
            ids, mask, label = ids.to(DEVICE), mask.to(DEVICE), label.to(DEVICE)
            out = model(input_ids=ids, attention_mask=mask, labels=label)
            total_val_loss += out.loss.item()
            pred = torch.argmax(out.logits, dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
    val_acc = correct / total
    val_loss = total_val_loss / len(loader)
    return val_loss, val_acc


# ===================== 训练（按步数验证 + 显示 loss） =====================
print("开始训练…")
global_step = 0

for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for batch in pbar:
        ids, mask, label = [x.to(DEVICE) for x in batch]
        optimizer.zero_grad()

        outputs = model(input_ids=ids, attention_mask=mask, labels=label)
        train_loss = outputs.loss.item()

        outputs.loss.backward()
        optimizer.step()
        scheduler.step()

        global_step += 1
        pbar.set_postfix(loss=train_loss, step=global_step)

        # ===================== 关键：每 eval_steps 步验证一次 + 显示 loss =====================
        if global_step % args.eval_steps == 0:
            val_loss, val_acc = evaluate(val_loader)
            print(f"\n===== 验证结果 | step={global_step} =====")
            print(f"训练 loss: {train_loss:.4f}")
            print(f"验证 loss: {val_loss:.4f}")
            print(f"验证准确率: {val_acc:.2%}")
            print("========================================\n")
            model.train()  # 切回训练

# ===================== 最后测试集评估 =====================
test_loss, test_acc = evaluate(test_loader)
print(f"\n测试集结果：")
print(f"测试 loss: {test_loss:.4f}")
print(f"测试准确率: {test_acc:.2%}")
print("\n训练完成！")