# bert lora 
bert-base-chinese lora 微调工程

## 数据清洗命令示例
```shell
python data_clean.py \
  --raw_csv data/raw_dataset.csv \
  --cleaned_csv data/cleaned_dataset.csv \
  --text_title sentence \
  --label_title label
```

## 训练命令示例
```shell
python train_lora.py \
  --train_csv data/cleaned_train_dataset.csv \
  --val_csv data/cleaned_valid_dataset.csv \
  --test_csv data/cleaned_test_dataset.csv \
  --eval_steps 1000 \
  --model_name_or_path /your/model/name/or/path \
  --max_len 128 \
  --batch_size 16 \
  --epochs 1 \
  --learning_rate 5e-5 \
  --text_title sentence \
  --label_title label \
  --target_modules query,key,value \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.01 \
  --logging_steps 50 \
  --output_dir output/bert-chinese-lora-sentiment \
  --warmup_ratio 0.1
```

## 推理命令示例
```shell
python predict.py \
  --model_name_or_path google-bert/bert-base-chinese \
  --lora_path output/bert-chinese-lora-sentiment \
  --port 8080
```

## 推送至 huggingface 脚本
```shell
# 登陆
hf auth login
# 执行推送脚本
python push_to_huggingface.py \
  --huggingface_username your_username \
  --model_repo_name your_model_repo \
  --lora_path output/bert-chinese-lora-sentiment
```

> https://huggingface.co/StarryXCN/bert-chinese-lora-sentiment