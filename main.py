from fastapi import FastAPI, status
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from static.utils import Model, decode_image, encode_image_to_base64


STATIC_PATH: str = "static"
VERSION: str = "0.0.1"

model: Model = Model()
model.setup()


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
    return JSONResponse({
        "statusText" : "Root Endpoint of Depth Inference API",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.get("/version")
async def get_version():
    return JSONResponse({
        "statusText" : "Version Fetch Successful",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.get("/infer")
async def get_infer():
    return JSONResponse({
        "statusText" : "Inference Endpoint",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.post("/infer")
async def post_infer(image: Image):
    _, image = decode_image(image.imageData)

    image = model.infer(image=image)
    imageData = encode_image_to_base64(image=image)

    return JSONResponse({
        "statusText" : "Depth Inference Complete",
        "statusCode" : status.HTTP_200_OK,
        "imageData" : imageData,
    })
