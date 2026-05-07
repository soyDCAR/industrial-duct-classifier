# syntax=docker/dockerfile:1
#
# ── Clasificador de Ductos ─────────────────────────────────────────
#
# Build CPU (por defecto):
#   docker build -t ductos .
#
# Build GPU (requiere driver NVIDIA y nvidia-container-toolkit):
#   docker build --build-arg CUDA=1 -t ductos-gpu .
#
# Ejecutar predicción sobre una imagen:
#   docker run --rm -v $(pwd)/modelo_ductos_multitarea_efnet.pth:/app/modelo.pth \
#              -v $(pwd)/class_mapping.json:/app/class_mapping.json \
#              ductos python predict.py imagen.jpg --model modelo.pth
#
# Ejecutar demo Gradio (puerto 7860):
#   docker run --rm -p 7860:7860 \
#              -v $(pwd)/modelo_ductos_multitarea_efnet.pth:/app/modelo.pth \
#              -v $(pwd)/class_mapping.json:/app/class_mapping.json \
#              ductos python app.py
# ──────────────────────────────────────────────────────────────────

ARG CUDA=0

# ── Stage CPU: Python slim + torch CPU wheels ──────────────────────
FROM python:3.10-slim AS stage-0

RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cpu

# ── Stage GPU: imagen oficial PyTorch con CUDA 11.8 ────────────────
# torch, torchvision y torchaudio ya vienen incluidos en esta imagen
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime AS stage-1

# ── Stage final: selecciona CPU (0) o GPU (1) ──────────────────────
# El truco: FROM stage-${CUDA} elige entre stage-0 y stage-1
FROM stage-${CUDA} AS runtime

# Metadatos del proyecto
LABEL maintainer="Dilan Acosta"
LABEL description="Clasificador multitarea de ductos — EfficientNet-B0"
LABEL version="1.0"

WORKDIR /app

# Instalar dependencias del proyecto (todo excepto torch, ya instalado)
# Copiamos requirements.txt primero para aprovechar la caché de Docker:
# si el archivo no cambia, Docker reutiliza esta capa en builds futuros
COPY requirements.txt .
RUN pip install --no-cache-dir \
    Pillow>=9.5.0 \
    opencv-python-headless>=4.7.0 \
    scikit-learn>=1.3.0 \
    pandas>=2.0.0 \
    scipy>=1.11.0 \
    matplotlib>=3.7.0 \
    tqdm>=4.65.0 \
    transformers>=4.35.0 \
    gradio>=4.0.0

# Copiar solo el código fuente (el modelo se monta como volumen)
COPY model.py predict.py train.py ./

COPY app.py .

# Puerto para la demo Gradio
EXPOSE 7860

# Por defecto muestra la ayuda; el usuario sobreescribe con su comando
CMD ["python", "predict.py", "--help"]
