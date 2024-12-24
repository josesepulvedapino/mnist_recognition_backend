import numpy as np

def predict_digit(model, processed_image):
    predictions = model.predict(processed_image)
    predicted_digit = np.argmax(predictions[0])
    probabilities = predictions[0]
    return predicted_digit, probabilities