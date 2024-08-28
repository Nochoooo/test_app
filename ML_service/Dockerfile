# Используем образ Python 3.10
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /ML_service

# Копируем файл с зависимостями
COPY ./requirements.txt /ML_service/requirements.txt

# Устанавливаем зависимости из requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем конкретную версию kafka-python для устранения ошибки
# RUN pip install --no-cache-dir kafka-python==1.4.7

# Применяем патч: заменяем 'async' на 'is_async' в проблемном файле (если требуется)
# RUN sed -i 's/self.async/self.is_async/g' /usr/local/lib/python3.10/site-packages/kafka/producer/simple.py

# Копируем весь код в контейнер
COPY . /ML_service


CMD ["uvicorn", "ML_service:model", "--host", "0.0.0.0", "--port", "8001"]
# Указываем команду запуска
# CMD ["python", "model.py"]

