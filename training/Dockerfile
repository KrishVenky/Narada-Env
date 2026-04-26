FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# System deps
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Python deps — unsloth last so it can detect the CUDA version at build time
RUN pip install --no-cache-dir \
    "trl>=0.9.0" \
    "peft>=0.10.0" \
    "accelerate>=0.30.0" \
    "bitsandbytes>=0.43.0" \
    "websockets>=12.0" \
    "pydantic>=2.7.0" \
    "openai>=1.30.0" \
    "nest_asyncio>=1.6.0" \
    "datasets>=2.18.0" \
    "transformers>=4.40.0" \
    "huggingface_hub>=0.22.0" \
    "matplotlib>=3.7.0"

RUN pip install --no-cache-dir unsloth_zoo && \
    pip install --no-cache-dir \
    "unsloth @ git+https://github.com/unslothai/unsloth.git"

WORKDIR /app
COPY train.py .

# Required: HF_TOKEN, HF_PUSH_REPO
# Optional: BASE_MODEL, ENV_URL, LORA_RANK
CMD ["python", "train.py"]
