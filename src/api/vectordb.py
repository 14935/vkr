from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

class VStore:
    def __init__(self, d, root='vectordata'):
        self.qc = QdrantClient(path=root)
        self.d = d

    def ensure(self, name: str):
        ex = [n.name for n in self.qc.get_collections().collections]
        if name not in ex:
            self.qc.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=self.d, distance=Distance.COSINE)
            )

    def put(self, name: str, slices, vectors):
        pts = [
            PointStruct(id=str(uuid4()), vector=v, payload={'text': s})
            for s, v in zip(slices, vectors)
        ]
        self.qc.upsert(collection_name=name, points=pts)

    def fetch(self, name: str, vec, lim: int = 10):
        out = self.qc.search(collection_name=name, query_vector=vec, limit=lim)
        return [h.payload['text'] for h in out if 'text' in h.payload]

    def names(self):
        return [n.name for n in self.qc.get_collections().collections]
