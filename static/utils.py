import io
import cv2
import onnx
import base64
import numpy as np
import onnxruntime as ort

from PIL import Image


class Processor:
    def __init__(self) -> None:
         pass

    @staticmethod
    def decode_image(data: str) -> np.ndarray:
        return np.array(Image.open(io.BytesIO(data)).convert("RGB"))

    @staticmethod
    def encode_image_to_base64(header: str = "data:image/png;base64", image: np.ndarray = None) -> str:
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
        _, imageData = cv2.imencode(".png", image)
        imageData = base64.b64encode(imageData)
        imageData = str(imageData).replace("b'", "").replace("'", "")
        imageData = header + "," + imageData
        return imageData
    
    @staticmethod  
    def write_to_temp(image: np.ndarray, filename) -> None:
        cv2.imwrite(filename, cv2.cvtColor(src=image, code=cv2.COLOR_RGB2BGR))


class Model(object):
    def __init__(self) -> None:
        ort.set_default_logger_severity(3)

        self.ort_session = None
        self.size: int = 256
        self.mean: list = [0.5, 0.5, 0.5]
        self.std: list = [0.5, 0.5, 0.5]

        model = onnx.load("static/model.onnx")
        onnx.checker.check_model(model)
        self.ort_session = ort.InferenceSession(
            "static/model.onnx", providers=["AzureExecutionProvider", "CPUExecutionProvider"]
        )

    async def infer(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape

        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
        image = image / 255
        image = cv2.resize(
            src=image, dsize=(self.size, self.size), interpolation=cv2.INTER_CUBIC
        ).transpose(2, 0, 1)
        for i in range(image.shape[0]):
            image[i, :, :] = (image[i, :, :] - self.mean[i]) / self.std[i]
        image = np.expand_dims(image, axis=0)
        input = {self.ort_session.get_inputs()[0].name: image.astype("float32")}
        result = self.ort_session.run(None, input)
        result = result[0].transpose(1, 2, 0)
        result = cv2.applyColorMap(
            src=cv2.convertScaleAbs(src=result, alpha=0.2), colormap=cv2.COLORMAP_JET
        )
        return cv2.resize(src=result, dsize=(w, h), interpolation=cv2.INTER_CUBIC)

model = Model()