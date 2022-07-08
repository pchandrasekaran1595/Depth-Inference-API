import os
import io
import cv2
import onnx
import base64
import numpy as np
import onnxruntime as ort

from PIL import Image

STATIC_PATH = "static"

class CFG(object):
    def __init__(self) -> None:
        self.ort_session = None
        self.size: int = 256
        self.mean: list = [0.5, 0.5, 0.5]
        self.std: list  = [0.5, 0.5, 0.5]
        self.path: str = os.path.join(STATIC_PATH, "model.onnx")
        ort.set_default_logger_severity(3)
    
    def setup(self) -> None:
        model = onnx.load(self.path)
        onnx.checker.check_model(model)
        self.ort_session = ort.InferenceSession(self.path)
    
    def infer(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape

        image = image / 255
        image = cv2.resize(src=image, dsize=(self.size, self.size), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
        for i in range(image.shape[0]):
            image[i, :, :] = (image[i, :, :] - self.mean[i]) / self.std[i]
        image = np.expand_dims(image, axis=0)
        input = {self.ort_session.get_inputs()[0].name : image.astype("float32")}
        result = self.ort_session.run(None, input)
        result = result[0].transpose(1, 2, 0)
        result = cv2.applyColorMap(src=cv2.convertScaleAbs(src=result, alpha=0.8), colormap=cv2.COLORMAP_JET)
        return cv2.resize(src=result, dsize=(w, h), interpolation=cv2.INTER_AREA)


def decode_image(imageData) -> np.ndarray:
    header, imageData = imageData.split(",")[0], imageData.split(",")[1]
    image = np.array(Image.open(io.BytesIO(base64.b64decode(imageData))))
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGRA2RGB)
    return header, image


def encode_image_to_base64(header: str = "data:image/png;base64", image: np.ndarray = None) -> str:
    assert image is not None, "Image is None"
    _, imageData = cv2.imencode(".jpeg", image)
    imageData = base64.b64encode(imageData)
    imageData = str(imageData).replace("b'", "").replace("'", "")
    imageData = header + "," + imageData
    return imageData
