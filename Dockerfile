FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src
COPY configs /app/configs
COPY docs /app/docs
COPY tests /app/tests
COPY scripts /app/scripts
RUN pip install --no-cache-dir -e ".[dev]"
ENTRYPOINT ["veilscan"]
CMD ["--help"]
