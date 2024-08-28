import os
import numpy as np
import keras
import keras_nlp
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from kafka import KafkaProducer, KafkaConsumer
import json

app = FastAPI()

# Модель для получения данных из запроса
class PredictRequest(BaseModel):
    text: str

class WeightedKappa(keras.metrics.Metric):
    @keras.utils.register_keras_serializable()
    def __init__(self, num_classes=6, epsilon=1e-6, name='weighted_kappa', **kwargs):
        super(WeightedKappa, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.epsilon = epsilon
        label_vec = keras.ops.arange(num_classes, dtype=keras.backend.floatx())
        self.row_label_vec = keras.ops.reshape(label_vec, [1, num_classes])
        self.col_label_vec = keras.ops.reshape(label_vec, [num_classes, 1])
        col_mat = keras.ops.tile(self.col_label_vec, [1, num_classes])
        row_mat = keras.ops.tile(self.row_label_vec, [num_classes, 1])
        self.weight_mat = (col_mat - row_mat) ** 2
        self.numerator = self.add_weight(name="numerator", initializer="zeros")
        self.denominator = self.add_weight(name="denominator", initializer="zeros")
        self.o_sum = self.add_weight(name='o_sum', initializer='zeros')
        self.e_sum = self.add_weight(name='e_sum', initializer='zeros')

    def update_state(self, y_true, y_pred, **args):
        y_true = keras.ops.one_hot(keras.ops.sum(y_true, axis=-1) - 1, 6)
        y_pred = keras.ops.one_hot(
            keras.ops.sum(keras.ops.cast(y_pred > 0.5, dtype="int8"), axis=-1) - 1, 6
        )
        y_true = keras.ops.cast(y_true, dtype=self.col_label_vec.dtype)
        y_pred = keras.ops.cast(y_pred, dtype=self.weight_mat.dtype)
        batch_size = keras.ops.shape(y_true)[0]
        cat_labels = keras.ops.matmul(y_true, self.col_label_vec)
        cat_label_mat = keras.ops.tile(cat_labels, [1, self.num_classes])
        row_label_mat = keras.ops.tile(self.row_label_vec, [batch_size, 1])
        weight = (cat_label_mat - row_label_mat) ** 2
        self.numerator.assign_add(keras.ops.sum(weight * y_pred))
        label_dist = keras.ops.sum(y_true, axis=0, keepdims=True)
        pred_dist = keras.ops.sum(y_pred, axis=0, keepdims=True)
        w_pred_dist = keras.ops.matmul(
            self.weight_mat, keras.ops.transpose(pred_dist, [1, 0])
        )
        self.denominator.assign_add(
            keras.ops.sum(keras.ops.matmul(label_dist, w_pred_dist))
        )
        self.o_sum.assign_add(keras.ops.sum(y_pred))
        self.e_sum.assign_add(keras.ops.sum(
            keras.ops.matmul(keras.ops.transpose(label_dist, [1, 0]), pred_dist)
        ))

    def result(self):
        return 1.0 - (
                keras.ops.divide_no_nan(self.numerator, self.denominator)
                * keras.ops.divide_no_nan(self.e_sum, self.o_sum)
        )

    def get_config(self):
        config = super(WeightedKappa, self).get_config()
        config.update({'num_classes': self.num_classes, 'epsilon': self.epsilon})
        return config

    def reset_state(self):
        self.numerator.assign(0)
        self.denominator.assign(0)
        self.o_sum.assign(0)
        self.e_sum.assign(0)

class EssayModel:
    def __init__(self):
        # Ensure the model path is correct within the container
        model_path = os.path.join('/ML_service', 'my_model.keras')
        self.model = keras.models.load_model(model_path, custom_objects={'WeightedKappa': WeightedKappa})
        self.preprocessor = keras_nlp.models.DebertaV3Preprocessor.from_preset(
            preset="deberta_v3_extra_small_en", 
            sequence_length=512
        )

    def predict_score(self, text):
        ds = self.preprocessor(text)
        test_preds = self.model.predict(ds, verbose=0)
        score = float(np.sum((test_preds > 0.5).astype(int)).clip(1, 6))
        return score

model = EssayModel()

# Инициализация Kafka
producer = KafkaProducer(bootstrap_servers='kafka:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'))
consumer = KafkaConsumer('predict_requests', bootstrap_servers='kafka:9092', value_deserializer=lambda v: json.loads(v.decode('utf-8')))

@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        # Отправка текста в Kafka
        producer.send('predict_requests', {'text': request.text})

        # Чтение сообщения с результатом предсказания
        for msg in consumer:
            result = msg.value
            if 'score' in result:
                return {'score': result['score']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
