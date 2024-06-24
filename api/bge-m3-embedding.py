from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel(
    "BAAI/bge-m3", use_fp16=True
)  # Setting use_fp16 to True speeds up computation with a slight performance degradation

from typing import Union

from fastapi import FastAPI

app = FastAPI()


from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    input: Union[list[str], str, None] = None
    model: str = "BAAI/bge-m3"
    encoding_format: str = "float"


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/embeddings")
async def embedding(request: EmbeddingRequest):
    if request.input is None or type(request.input) == str or len(request.input) == 0:
        return {
            "object": "list",
            "data": [],
            "model": "BAAI/bge-m3",
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0,
            },
        }
    resp = [
        {
            "object": "embedding",
            "index": 0,
            "embedding": model.encode(st, batch_size=6, max_length=8192)[
                "dense_vecs"
            ].tolist(),
        }
        for st in request.input
    ]
    return {
        "object": "list",
        "data": resp,
        "model": "BAAI/bge-m3",
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.post("/legacy/embeddings")
async def legacy_embedding(request: EmbeddingRequest):
    if request.input is None or type(request.input) == str or len(request.input) == 0:
        return {
            "object": "list",
            "data": [],
            "model": "GAAI/bge-m3",
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0,
            },
        }
    resp = [
        {
            "object": "embedding",
            "index": 0,
            "embedding": model.encode(st, batch_size=6, max_length=8192)[
                "dense_vecs"
            ].tolist(),
        }
        for st in request.input
    ]
    return {
        "object": "list",
        "data": resp,
        "model": "GAAI/bge-m3",
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
    }
