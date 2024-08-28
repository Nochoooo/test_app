FROM python:3.10-slim

WORKDIR /ui_service
COPY ./requirements.txt /ui_service/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем конкретную версию kafka-python для устранения ошибки
# RUN pip install --no-cache-dir kafka-python==1.4.7

# Применяем патч: заменяем 'async' на 'is_async' в проблемном файле (если требуется)
# RUN sed -i 's/self.async/self.is_async/g' /usr/local/lib/python3.10/site-packages/kafka/producer/simple.py

COPY . /ui_service

CMD ["uvicorn", "ui_service:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["python", "app.py"]