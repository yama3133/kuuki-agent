"""python -m web [--port 8000]"""

import argparse

import uvicorn

parser = argparse.ArgumentParser(prog="python -m web")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()
uvicorn.run("web.app:app", host=args.host, port=args.port)
