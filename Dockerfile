FROM python:3.12.1
WORKDIR /analyst
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
COPY . .
RUN python -m venv /ven
ENV PATH="/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt
VOLUME /analyst/config
EXPOSE 8000


CMD ["python --main"]