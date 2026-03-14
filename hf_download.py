

from huggingface_hub import snapshot_download
import os

def main():
    repo_id = "Salesforce/blip-vqa-base"
 
    os.environ["HF_HOME"] = r"E:\hf_cache"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    local_dir = "hf_tmp_blip_vqa"
    print("Downloading:", repo_id)
    print("HF_HOME    :", os.environ["HF_HOME"])

    path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        resume_download=True,
        # 只下载必要文件：配置+分词器+safetensors 权重
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "tokenizer.json",
            "*.safetensors",
        ],
        # 避免把大模型的 .bin 也拉下来
        ignore_patterns=["*.bin", "*.h5", "*.ot", "*.msgpack"],
    )
    print("✅ Done. Snapshot path:", path)

if __name__ == "__main__":
    main()