from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

class VStore:
    def __init__(self, d, root='vectordata'):
        self.qc = QdrantClient(path=root)
        self.d = d

    def ensure(self, name):
        ex = [n.name for n in self.qc.get_collections().collections]
        if name not in ex:
            self.qc.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=self.d, distance=Distance.COSINE)
            )

    def put(self, name, slices, vectors):
        pts = [
            PointStruct(id=f"{name}_{i}", vector=v, payload={'text': s})
            for i, (s, v) in enumerate(zip(slices, vectors))
        ]
        self.qc.upsert(collection_name=name, points=pts)

    def fetch(self, name, vec, lim=10):
        out = self.qc.search(collection_name=name, query_vector=vec, limit=lim)
        return [h.payload['text'] for h in out if 'text' in h.payload]

    def names(self):
        return [n.name for n in self.qc.get_collections().collections]
