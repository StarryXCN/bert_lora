# bert_lora
bert lora 微调

训练命令示例
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
	--output_dir output/bert_lora_sentiment \
	--warmup_ratio 0.1
```