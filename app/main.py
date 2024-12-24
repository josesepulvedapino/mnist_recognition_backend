from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from app.model.predictor import predict_digit
from app.utils.image_utils import preprocess_image
import os

app = FastAPI(title="MNIST Digit Recognition API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mnist-recognition-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Obtener ruta absoluta al modelo
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'mnist_model.h5')

# Cargar modelo
model = tf.keras.models.load_model(model_path)

@app.get("/")
def read_root():
    return {"message": "API de Reconocimiento de Dígitos"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        processed_image = preprocess_image(contents)
        prediction, probabilities = predict_digit(model, processed_image)
        
        return {
            "success": True,
            "predicted_digit": int(prediction),
            "probabilities": {
                str(i): float(prob) 
                for i, prob in enumerate(probabilities)
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }