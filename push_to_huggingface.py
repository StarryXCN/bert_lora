from huggingface_hub import HfApi
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--huggingface_username", required=True, type=str, help="huggingface用户名")
parser.add_argument("--model_repo_name", required=True, type=str, help="模型库名称")
parser.add_argument("--lora_path", required=True, type=str, help="lora路径")
args = parser.parse_args()

api = HfApi()

# 上传整个文件夹
api.upload_folder(
    folder_path=args.lora_path,
    repo_id=f"{args.huggingface_username}/{args.model_repo_name}",
    repo_type="model"
)

print(f"上传成功！访问地址：https://huggingface.co/{args.huggingface_username}/{args.model_repo_name}")