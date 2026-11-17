FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# main while iterating; switch to a release tag (for example v1.0.0) once the Space is stable
ARG QUERION_REF=main
RUN git clone --depth 1 --branch ${QUERION_REF} https://github.com/AmanSingh-404/QUERION.git .
RUN pip install --no-cache-dir "torch==$(cat docker/torch-version.txt)" \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r docker/requirements.txt

ENV QUERION_DEVICE=cpu \
    QUERION_INT8=1 \
    QUERION_SAMPLES=4 \
    QUERION_ORIGINS=http://localhost:3000 \
    PORT=7860
EXPOSE 7860
CMD ["python", "-m", "api.main"]