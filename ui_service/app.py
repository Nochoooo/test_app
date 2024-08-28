from fastapi import FastAPI, Request, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from loguru import logger
from kafka import KafkaProducer
from kafka.errors import KafkaError
from werkzeug.utils import secure_filename
import os
import json
from database import Database

app = FastAPI()

# Инициализация Kafka producer
kafka_server = os.environ.get('KAFKA_SERVER', 'kafka:9092')
kafka_topic = os.environ.get('KAFKA_TOPIC', 'predict_requests')
producer = KafkaProducer(bootstrap_servers=kafka_server, value_serializer=lambda v: json.dumps(v).encode('utf-8'))

# Путь к директории для загруженных файлов
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Инициализация объектов базы данных
database = Database()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/")
async def handle_form(
    author: str = Form(...), 
    essay_file: UploadFile = Form(...)
):
    database.create_essays_table_if_not_exists()
    
    if not essay_file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Файл должен иметь формат .txt")
    
    content = await essay_file.read()
    
    try:
        content = content.decode('utf-8')
    except UnicodeDecodeError as e:
        logger.error(f"Ошибка декодирования файла: {e}")
        raise HTTPException(status_code=400, detail="Ошибка декодирования файла.")
    
    if not content:
        logger.error("Файл пустой или не удалось прочитать содержимое файла.")
        raise HTTPException(status_code=400, detail="Файл пустой или не удалось прочитать содержимое файла.")
    
    # Сохранение файла
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(essay_file.filename))
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"Файл успешно сохранен по пути: {file_path}")
    
    # Отправка данных в Kafka
    try:
        producer.send(kafka_topic, {'author': author, 'file_path': file_path, 'content': content})
    except KafkaError as e:
        logger.error(f"Ошибка отправки сообщения в Kafka: {e}")
        raise HTTPException(status_code=500, detail="Ошибка отправки данных на ML сервис.")
    
    return RedirectResponse(url="/", status_code=303)

@app.get("/view", response_class=HTMLResponse)
async def view(request: Request):
    essays = database.get_all_essays()
    if essays is None:
        raise HTTPException(status_code=500, detail="Ошибка при получении данных из базы.")
    return templates.TemplateResponse("view.html", {"request": request, "essays": essays})

@app.post("/delete")
async def delete(data: dict):
    ids_to_delete = data['ids']
    ids_to_delete = [int(i) for i in ids_to_delete]
    if database.delete_essays(ids_to_delete):
        return {}
    raise HTTPException(status_code=500, detail="Ошибка при удалении записей.")

@app.post("/update_score")
async def update_score(data: dict):
    essay_id = data['id']
    new_score = data['score']
    try:
        new_score = float(new_score)
        if not (1 <= new_score <= 6):
            raise ValueError('Недопустимая оценка.')
    except ValueError as e:
        logger.error(f"Ошибка проверки: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    if database.update_essay_score(essay_id, new_score):
        return {}
    raise HTTPException(status_code=500, detail="Ошибка при обновлении оценки.")

# Точка входа для запуска приложения
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000, debug=True)

