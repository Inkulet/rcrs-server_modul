# Я собираю лёгкий образ агентского модуля RCRS.
# Ядро симулятора (rcrs-server, Java) в образ не входит — агент
# подключается к нему по TCP (--host/--port).
FROM python:3.11-slim

WORKDIR /app

# сначала зависимости — слой кэшируется отдельно от кода
COPY rcrs_module/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rcrs_module/ .

# main.py сам добавляет src/ в sys.path
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
