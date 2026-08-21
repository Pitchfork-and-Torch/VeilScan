FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src
COPY configs /app/configs
COPY docs /app/docs
RUN pip install --no-cache-dir -e .
ENTRYPOINT ["veilscan"]
CMD ["--help"]
