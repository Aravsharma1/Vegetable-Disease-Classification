import numpy as np
from fastapi import FastAPI,File, UploadFile
import uvicorn
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image

app = FastAPI()



@app.get("/ping") # this is how you specify an end point
async def ping():
    return "Hello, the server is working!"

production_model = tf.keras.models.load_model("../../saved_models/2")
beta_model = tf.keras.models.load_model("../../saved_models/3") # try and apply docker onto this
CLASS_NAMES = ["Early Bright", "Late Bright", "Healthy"]
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
        file:UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = production_model.predict(img_batch) # this is an array
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    } # this is a dictionary


    # np.argmax returns the index of the array that consist of the maximum element of the array
# watch something about async io tutorial to further understand

if __name__ == "__main__":
    uvicorn.run(app, host = 'localhost', port =8000)

