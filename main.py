import os
import sys
import json

from typing import Union

from sanic import Sanic
from sanic.request import Request
from sanic.exceptions import SanicException
from sanic.response import html, file, JSONResponse

from static.backend.utils import model, Processor


STATIC_PATH: str = "static"
PORT: int = 9090

app = Sanic("Depth-Inference-API")
app.static("static", "static")

if not os.path.exists("TEMP"):
    os.makedirs("TEMP")


@app.route("/favicon.ico", methods=["GET"])
async def favicon(request: Request):
    return await file(location=f"{STATIC_PATH}/web/favicon.ico", status=200, mime_type="image/x-icon")


@app.route("/", methods=["GET"])
async def root(request: Request) -> html:
    with open(f"{STATIC_PATH}/web/index.html", "r") as fp:
        html_content = fp.read()
    return html(body=html_content, status=200)


@app.route("/help", methods=["GET"])
async def help(request: Request) -> html:
    with open(f"{STATIC_PATH}/web/help.html", "r") as fp:
        html_content = fp.read()
    return html(body=html_content, status=200)


@app.route("/login", methods=["POST"])
async def login(request: Request) -> JSONResponse:
    try:
        await request.receive_body()
    except:
        raise SanicException(
            message={
                "statusText" : "Error in request body"
            }, 
            status_code=400
        )

    body = json.loads(request.body.decode("ascii"))
    username = body["username"]
    password = body["password"]
    timestamp = body["timestamp"]

    if not os.path.exists("logs"):
        os.makedirs("logs")

    with open("logs/logs.txt", "a") as fp:
        fp.write(f"login attempt at {timestamp} - {username}, {password}\n")

    return JSONResponse(
        body={
            "statusText": "Login Successful",
        },
        status=201
    )


@app.route("/depth", methods=["GET", "POST"])
async def resize(request: Request) -> Union[JSONResponse, file]:
    '''
    BASH

    curl -X GET -L "http://localhost:9090/depth"
    curl -X GET -L "http://localhost:9090/depth?rtype=json"
    curl -X POST -L "http://localhost:9090/depth" -F file=@"/C:/Users/user/IMG.png" --output C:/Users/user/Downloads/Depth.png
    curl -X POST -L "http://localhost:9090/depth?rtype=json" -F file=@"/C:/Users/user/IMG.png" --output C:/Users/user/Downloads/Depth.json

    '''
    rtype: str = "file"

    if request.method == "GET":
        if len(request.args) == 0:
            return JSONResponse(
                body={
                    "statusText": "Depth Inference Endpoint",
                },
                status=200,
            )
        
        if "rtype" in request.args:
            rtype = request.args.get("rtype")

        return JSONResponse(
            body={
                "statusText": f"Depth Inference Endpoint (rtype={rtype})",
            },
            status=200,
        )

    elif request.method == "POST":
        if "rtype" in request.args:
            rtype = request.args.get("rtype")
        
        image = model.infer(Processor.decode_image(request.files.get("file").body))

        if rtype == "json":
            return JSONResponse(
                body={
                    "statusText": "Image Resize Successful",
                    "imageData": Processor.encode_image_to_base64(image=image),
                },
                status=201,
            )
        elif rtype == "file":
            Processor.write_to_temp(image)
            return await file(location="TEMP/temp.png", status=201, mime_type="image/*")
        else:
            raise SanicException(
                message={
                    "statusText" : "Invalid return type"
                },
                status_code=400
            )
        

if __name__ == "__main__":
    args_1: tuple = ("-m", "--mode")
    args_2: tuple = ("-p", "--port")
    args_3: tuple = ("-w", "--workers")

    mode: str = "local-machine"
    port: int = 9090
    workers: int = 1

    if args_1[0] in sys.argv:
        mode = sys.argv[sys.argv.index(args_1[0]) + 1]
    if args_1[1] in sys.argv:
        mode = sys.argv[sys.argv.index(args_1[1]) + 1]
    
    if args_2[0] in sys.argv:
        port = int(sys.argv[sys.argv.index(args_2[0]) + 1])
    if args_2[1] in sys.argv:
        port = int(sys.argv[sys.argv.index(args_2[1]) + 1])

    if args_3[0] in sys.argv:
        workers = int(sys.argv[sys.argv.index(args_3[0]) + 1])
    if args_3[1] in sys.argv:
        workers = int(sys.argv[sys.argv.index(args_3[1]) + 1])

    if mode == "local-machine":
        app.run(host="localhost", port=port, dev=True, workers=workers)

    elif mode == "local":
        app.run(host="0.0.0.0", port=port, dev=True, workers=workers)

    elif mode == "render":
        app.run(host="0.0.0.0", port=port, single_process=True, access_log=True)

    elif mode == "prod":
        app.run(host="0.0.0.0", port=port, dev=False, workers=workers, access_log=True)

    else:
        raise ValueError("Invalid Mode")