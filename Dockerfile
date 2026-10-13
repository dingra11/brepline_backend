FROM python:3.11-slim
# fontconfig + fonts: CadQuery's text() needs at least one system font (python:3.11-slim has none)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglu1-mesa libxrender1 libxext6 libsm6 \
    fontconfig fonts-dejavu-core fonts-liberation && fc-cache -f && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# fail the BUILD (not a user's job) if text rendering is broken
RUN python -c "import cadquery as cq; r = cq.Workplane('XY').text('HELLO', 10, 2, combine=True); assert r.val().Volume() > 0; print('text ok')"
COPY . .
RUN chown -R user:user /home/user/app
USER user
ENV OUT_DIR=/tmp/cad_out DATA_DIR=/tmp/cad_data PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860}"]
