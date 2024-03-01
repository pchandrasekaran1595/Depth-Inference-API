import os
import sys

from typing import Union

from sanic import Sanic
from sanic.request import Request
from sanic.response import file, JSONResponse

from static.utils import model, Processor


STATIC_PATH: str = "static"
PORT: int = 9090

app = Sanic("Depth-Inference-API")
app.static("static", "static")

if not os.path.exists("TEMP"):
    os.makedirs("TEMP")


@app.route("/", methods=["GET"])
async def root(request: Request) -> JSONResponse:
    """
    BASH
        curl -X GET "<BASE_URL>" -s
    """
    return JSONResponse(
        body={
            "statusText": "Root Endpoint of Depth-Inference-API",
        },
        status=200,
    )


@app.route("/clean", methods=["GET"])
async def clean(request: Request) -> JSONResponse:
    """
    BASH
        curl -X GET "<BASE_URL>/clean" -s
    """
    if len(os.listdir("TEMP")) == 0:
        return JSONResponse(
            body={
                "statusText": "Temp Directory is already Clean",
            },
            status=200,
        )

    for filename in os.listdir("TEMP"):
        os.remove(f"TEMP/{filename}")

    return JSONResponse(
        body={
            "statusText": "Cleaned Temp Directory",
        },
        status=200,
    )


@app.route("/depth", methods=["GET", "POST"], name="Depth")
async def resize(request: Request) -> Union[JSONResponse, file]:
    """
    BASH
        curl -X GET -L "<BASE_URL>/depth"
        curl -X GET -L "<BASE_URL>/depth?rtype=json"
        curl -X POST -L "<BASE_URL>/depth" -F file=@"/<PATH>/<NAME>.<EXT>" -o <PATH>/<NAME>.png
        curl -X POST -L "<BASE_URL>/depth?rtype=json" -F file=@"/<PATH>/<NAME>.<EXT>" -o <PATH>/<NAME>.json
    """
    rtype: str = "file"

    if request.method == "GET":
        if "rtype" in request.args:
            rtype = request.args.get("rtype")

        return JSONResponse(
            body={
                "statusText": f"Depth Inference Endpoint (rtype={rtype})",
            },
            status=200,
        )

    elif request.method == "POST":
        if request.files.get("file", None) is None:
            return JSONResponse(
                body={"statusText": "Invalid Key Specified for file Upload"},
                status=400,
            )

        if "rtype" in request.args:
            rtype = request.args.get("rtype")

        filename: str = request.files.get("file").name

        image = await model.infer(
            Processor.decode_image(request.files.get("file").body)
        )

        if rtype == "json":
            return JSONResponse(
                body={
                    "statusText": "Image Resize Successful",
                    "imageData": Processor.encode_image_to_base64(image=image),
                },
                status=201,
            )
        elif rtype == "file":
            Processor.write_to_temp(image, f"TEMP/temp_{filename.split('.')[0]}.png")
            return await file(
                location=f"TEMP/temp_{filename.split('.')[0]}.png",
                status=201,
                mime_type="image/*",
            )
        else:
            return JSONResponse(
                body={"statusText": "Invalid Return Type Specified"}, status=400
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
