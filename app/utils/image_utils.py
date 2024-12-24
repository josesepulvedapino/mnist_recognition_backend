from PIL import Image
import numpy as np
import io

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array