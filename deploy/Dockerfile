FROM vllm/vllm-openai:nightly

# Install ASR packages on top of existing vLLM install.
# Both register custom model architectures with vLLM's ModelRegistry.
RUN pip install --no-cache-dir qwen-asr meralion-3-asr

# Serve from pre-downloaded HF cache (mounted at runtime).
ENV HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
