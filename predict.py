import torch
import argparse
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel
import uvicorn

# ==================== 命令行参数 ====================
parser = argparse.ArgumentParser(description="BERT-LoRA 情感分类 API")
parser.add_argument("--model_name_or_path", required=True, type=str, help="基础模型路径或ID")
parser.add_argument("--lora_path", required=True, type=str, help="LoRA权重路径")
parser.add_argument("--port", type=int, default=8000, help="服务端口")
args = parser.parse_args()

# ==================== 日志 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ==================== 设备 ====================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

logger.info(f"使用设备: {device}")
logger.info(f"基础模型: {args.model_name_or_path}")
logger.info(f"LoRA 路径: {args.lora_path}")

# ==================== 加载模型 ====================
tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

base_model = BertForSequenceClassification.from_pretrained(
    args.model_name_or_path,
    num_labels=2
)

model = PeftModel.from_pretrained(base_model, args.lora_path)
model.to(device)
model.eval()

logger.info("模型加载完成，服务启动成功！")

# ==================== FastAPI ====================
app = FastAPI(title="BERT-LoRA 情感分类 API")

# 单条请求
class SingleRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100, description="待分类文本")

# 批量请求
class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=32, description="批量文本列表")

# ==================== 单条预测 ====================
@app.post("/predict")
async def predict(req: SingleRequest):
    try:
        text = req.text.strip()
        inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            prob = torch.softmax(logits, dim=-1)[0]
            label = torch.argmax(prob).item()
            score = prob[label].item()

        return {
            "code": 200,
            "text": text,
            "label": label,
            "score": round(score, 4),
            "message": "success"
        }

    except Exception as e:
        logger.error(f"预测异常: {str(e)}")
        raise HTTPException(status_code=500, detail="预测失败")

# ==================== 批量预测 ====================
@app.post("/predict_batch")
async def predict_batch(req: BatchRequest):
    try:
        texts = [t.strip() for t in req.texts if t.strip()]
        if not texts:
            raise HTTPException(status_code=400, detail="有效文本不能为空")

        inputs = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        results = []
        for i, text in enumerate(texts):
            label = torch.argmax(probs[i]).item()
            score = probs[i][label].item()
            results.append({
                "text": text,
                "label": label,
                "score": round(score, 4)
            })

        return {
            "code": 200,
            "count": len(results),
            "results": results,
            "message": "success"
        }

    except Exception as e:
        logger.error(f"批量预测异常: {str(e)}")
        raise HTTPException(status_code=500, detail="批量预测失败")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="info"
    )