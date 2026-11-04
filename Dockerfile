# 单阶段构建:前端已预编译(见 dist/),构建过程不经过 node/npm,
# 内网/弱网环境更稳。想从源码完整重建前端,用 Dockerfile.from-source。
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# opencv 运行时依赖(扫描件 OCR 用;slim 镜像默认缺 libGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Python 依赖(优先清华镜像,失败回退官方源)。全部为预编译 wheel,无需 apt 装编译链;
# 如你本地的 Dockerfile 有 apt-get install gcc g++ cmake 一行,那是多余的,删掉可提速
COPY server/requirements.txt server/requirements.txt
RUN pip install --retries 5 --timeout 100 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple -r server/requirements.txt \
    || pip install --retries 5 --timeout 100 -r server/requirements.txt

# 后端代码 + 预置数据(已解析文档 + 预建 Milvus 索引 + 切分缓存)
COPY server ./server
# 预编译前端产物(随包交付,无需重新构建)
COPY dist ./dist

# 向量模型 bge-small-zh-v1.5 已随包预置(server/data/models,约 180MB),
# 构建与首次运行均无需联网下载;这里仅做存在性校验。
ENV HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
RUN python3 -c "import os,glob;     hits=glob.glob('server/data/models/*/snapshots/*/model_optimized.onnx');     assert hits, 'embedding model cache missing'; print('embedding model bundled:', hits[0])"
# 注:BGE 精排模型(bge-reranker-base,约1GB)默认不预置;在「知识库与文档解析→检索设置」
#     启用精排后,首次使用时自动从 hf-mirror 下载;模型不可用时自动回退到 RRF 排序。

ENV DATA_DIR=/app/server/data FRONTEND_DIST=/app/dist
EXPOSE 3000
# DeepSeek Key 可通过环境变量 DEEPSEEK_API_KEY 注入,或在「系统管理」页配置
CMD ["python3", "-m", "uvicorn", "app.main:app", "--app-dir", "server", \
     "--host", "0.0.0.0", "--port", "3000"]
