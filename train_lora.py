import csv
import logging

import torch
import pandas as pd
import argparse
from transformers import BertTokenizer, BertForSequenceClassification
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

logging.basicConfig(level=logging.INFO)

class Person:
    name: str
    age: int
    def __init__(self, name, age):
        self.name = name
        self.age = age

# ===================== 命令行参数 =====================
parser = argparse.ArgumentParser()
parser.add_argument("--train_csv", required=True, type=str, help="训练集CSV路径")
parser.add_argument("--val_csv", required=True, type=str, help="验证集CSV路径")
parser.add_argument("--test_csv", required=True, type=str, help="测试集CSV路径")
parser.add_argument("--eval_steps", required=True, type=int, help="每隔多少步验证一次")
parser.add_argument("--model_name_or_path", required=True, type=str, help="模型名称或路径")
parser.add_argument("--max_len", required=True, type=int, help="text最大长度")
parser.add_argument("--batch_size", required=True, type=int, help="批次大小")
parser.add_argument("--epochs", required=True, type=int, help="训练轮次")
parser.add_argument("--learning_rate", required=True, type=str, help="学习率")
parser.add_argument("--text_title", required=True, type=str, help="text 对应的 csv 标题")
parser.add_argument("--label_title", required=True, type=str, help="label 对应的 csv 标题")
parser.add_argument("--target_modules", required=True, type=str, help="要训练的模组")
parser.add_argument("--lora_rank", required=True, type=int, help="lora 秩")
parser.add_argument("--lora_alpha", required=True, type=int, help="lora 缩放系数")
parser.add_argument("--lora_dropout", required=True, type=float, help="lora 偏移")
parser.add_argument("--logging_steps", required=True, type=int, help="每多少次打印训练日志")
parser.add_argument("--output_dir", required=True, type=str, help="输出目录")
parser.add_argument("--warmup_ratio", required=True, type=float, help="预热系数")
args = parser.parse_args()

# ===================== 配置 =====================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
MODEL_NAME = args.model_name_or_path
MAX_LEN = args.max_len
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR = float(args.learning_rate)

# LoRA 配置
lora_config = LoraConfig(
    r = args.lora_rank,
    lora_alpha = args.lora_alpha,
    target_modules = args.target_modules.split(","),
    lora_dropout = args.lora_dropout,
    bias = "none",
    task_type = "SEQ_CLS"
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
        "text": [item[args.text_title] for item in dataset],
        "label": [int(item[args.label_title]) for item in dataset]
    }

train_df = pd.DataFrame(load_csv(args.train_csv))
val_df = pd.DataFrame(load_csv(args.val_csv))
test_df = pd.DataFrame(load_csv(args.test_csv))


# ===================== 编码函数 =====================
def encode(texts):
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
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
# 替换掉你原来的 线性 / 余弦 scheduler
scheduler = OneCycleLR(
    optimizer,
    max_lr=LR,           # 你设置的学习率 2e-5
    total_steps=total_steps,
    pct_start=args.warmup_ratio      # 前10%时间增长到最大LR
)


# ===================== 评估函数 =====================
def evaluate(loader, desc):
    model.eval()
    pbar = tqdm(loader, desc=desc)
    correct = 0
    total = 0
    total_val_loss = 0.0
    step = 0
    with torch.no_grad():
        for batch in pbar:
            ids, mask, label = [x.to(DEVICE) for x in batch]
            out = model(input_ids=ids, attention_mask=mask, labels=label)
            total_val_loss += out.loss.item()
            pred = torch.argmax(out.logits, dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
            step += 1
    val_acc = correct / total
    val_loss = total_val_loss / len(loader)
    return val_loss, val_acc


# ===================== 训练（按步数验证 + 显示 loss） =====================
logging.info("开始训练…")
logging.info(f"\n学习率 = {LR}")
global_step = 0

for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(train_loader, desc=f"轮次 {epoch + 1}")

    for batch in pbar:
        ids, mask, label = [x.to(DEVICE) for x in batch]
        optimizer.zero_grad()

        outputs = model(input_ids=ids, attention_mask=mask, labels=label)
        train_loss = outputs.loss.item()

        outputs.loss.backward()
        optimizer.step()
        scheduler.step()

        global_step += 1
        current_lr = optimizer.param_groups[0]['lr']

        if global_step % 50 == 0:
            pbar.write(f"{{训练 loss: {train_loss:.4f}, 训练学习率：{current_lr:.2e}}}")

        # ===================== 关键：每 eval_steps 步验证一次 + 显示 loss =====================
        if global_step % args.eval_steps == 0:
            val_loss, val_acc = evaluate(val_loader, "验证")
            pbar.write(f"验证结果：{{验证 loss: {val_loss:.4f}，验证准确率: {val_acc:.2%}}}")
            model.train()  # 切回训练

# ===================== 最后测试集评估 =====================
test_loss, test_acc = evaluate(test_loader, "测试")
logging.info(f"\n测试集结果：{{测试 loss: {test_loss:.4f}，测试准确率: {test_acc:.2%}}}")
logging.info("\n训练完成！")

# ===================== 保存 LoRA 模型 =====================
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
logging.info("LoRA 模型已保存！")