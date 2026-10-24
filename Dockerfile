# DeepSeek-V4.1-Flash-1017 CSF expert-streaming engine (v4) on the pinned LiL vLLM image.
# Weights, the expert pack, placement and caches stay on host paths; this image adds code only.
FROM ghcr.io/local-inference-lab/vllm@sha256:fd74019fc217a541060093d18427c36330df6ac5700651b55593dafdbb936920

# DeepSeek-V4.1-Flash-1017 tool-call encoder, copied unchanged from deepseek-ai/DeepSeek-V4.1-Flash-1017
# @ 1bae0e1088993b9b82de08f59de5be5a34e42170 (encoding/encoding.py, MIT).
COPY vendor/deepseek_v41_encoding_1017.py /opt/venv/lib/python3.12/site-packages/vllm/tokenizers/deepseek_v41_encoding.py
COPY csf_streaming /opt/csf/csf_streaming
COPY sitecustomize.py /opt/csf/sitecustomize.py
COPY patches /opt/csf/patches

ENV PYTHONPATH=/opt/csf:/opt/lil \
    CSF_EXT_BUILD_DIR=/opt/csf/ext-build \
    TORCH_CUDA_ARCH_LIST=12.0

RUN /opt/venv/bin/python /opt/csf/patches/deepseek_v41_images.py /opt/venv/lib/python3.12/site-packages/vllm/tokenizers/deepseek_v41.py \
 && /opt/venv/bin/python -m csf_streaming.prebuild \
 && rm -rf /opt/csf/csf_streaming/__pycache__
