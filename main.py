from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from static.utils import CFG, decode_image, encode_image_to_base64


STATIC_PATH = "static"
VERSION = "0.0.1"


class Image(BaseModel):
    imageData: str


origins = [
    "http://localhost:9091",
]


app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "statusText" : "Root Endpoint of Depth Inference API",
        "statusCode" : 200,
        "version" : VERSION,
    }


@app.get("/version")
async def get_version():
    return {
        "statusText" : "Version Fetch Successful",
        "statusCode" : 200,
        "version" : VERSION,
    }


@app.get("/infer")
async def get_infer():
    return {
        "statusText" : "Inference Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    }


@app.post("/infer")
async def post_infer(image: Image):
    _, image = decode_image(image.imageData)

    cfg = CFG()
    cfg.setup()
    image = cfg.infer(image=image)
    imageData = encode_image_to_base64(header="data:image/png;base64", image=image)

    return {
        "statusText" : "Depth Inference Complete",
        "statusCode" : 200,
        "imageData" : imageData,
    }
