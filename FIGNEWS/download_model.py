
# run shell command to download embeddings
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
prefix = "./model/"
import subprocess
models = [
            "BAAI/bge-large-en-v1.5",
]

for model in models:
    print(f"Downloading {model}")
    filename = model.split('/')[-1]
    os.system(f"huggingface-cli download --resume-download --local-dir-use-symlinks False {model} --local-dir {prefix}/{filename}")
# huggingface-cli download --resume-download --local-dir-use-symlinks False intfloat/multilingual-e5-large --local-dir multilingual-e5-large