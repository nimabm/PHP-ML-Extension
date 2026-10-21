# CPython with the reference libraries: NumPy, scikit-learn and PyTorch (CPU build).
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.5.4 scikit-learn==1.9.1 \
    && pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
