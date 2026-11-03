FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir flask pillow gunicorn
RUN mkdir -p models && python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/Indira002-02/NeuroVision-AI/releases/download/v1.0/bt_resnet50_model.pt', 'models/bt_resnet50_model.pt')"
COPY . .
EXPOSE 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "--workers", "1", "app:app"]
