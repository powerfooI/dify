from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class EmbeddingRequest(BaseModel):
    input: Union[list[str], str, None] = None
    model: str = "BAAI/bge-m3"
    encoding_format: str = "float"


class RerankRequest(BaseModel):
    model: str = "BAAI/bge-m3"
    query: str
    documents: list[str]
    top_n: Union[int, None] = None


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


@app.post("/legacy/embeddings") # Legacy endpoint for a mistake in the frontend
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


@app.post("/rerank")
async def rerank(request: RerankRequest):
    if request.documents is None or len(request.documents) == 0:
        return {
            "model": "BAAI/bge-m3",
            "query": request.query,
            "results": [],
        }
    scores = [
        model.compute_score(
            [request.query, st],
            batch_size=8,
            weights_for_different_modes=[0.4, 0.2, 0.4],
        )["colbert+sparse+dense"]
        for st in request.documents
    ]

    results = [
        {
            "index": i,
            "document": {
                "text": st,
            },
            "relevance_score": scores[i],
        }
        for i, st in enumerate(request.documents)
    ]
    return {
        "model": "BAAI/bge-m3",
        "query": request.query,
        "results": results,
    }


"""
# rerank request body 
{
    'model': 'BAAI/bge-m3', 
    'query': 'What is the capital of the United States?', 
    'documents': ['Carson City is the capital city of the American state of Nevada. At the 2010 United States Census, Carson City had a population of 55,274.', 
    'The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean that are a political division controlled by the United States. Its capital is Saipan.'], 
    'top_n': None
}
"""

"""
data = {
    "model": model_name,
    "query": query,
    "documents": docs,
    "top_n": top_n
}

try:
    response = post(str(URL(url) / 'rerank'), headers=headers, data=dumps(data), timeout=10)
    response.raise_for_status() 
    results = response.json()

    rerank_documents = []
    for result in results['results']:  
        rerank_document = RerankDocument(
            index=result['index'],
            text=result['document']['text'],
            score=result['relevance_score'],
        )
        if score_threshold is None or result['relevance_score'] >= score_threshold:
            rerank_documents.append(rerank_document)

    return RerankResult(model=model, docs=rerank_documents)
except httpx.HTTPStatusError as e:
    raise InvokeServerUnavailableError(str(e))  
"""
