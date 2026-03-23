from __future__ import annotations

from pathlib import Path
from typing import Iterable

import faiss
import numpy as np


class FaissEngine:
    def __init__(self, index_path: Path, dimension: int = 512) -> None:
        self.index_path = index_path
        self.dimension = dimension
        self.id_index = faiss.IndexIDMap2(faiss.IndexFlatL2(dimension))
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if self.index_path.exists():
            self.id_index = faiss.read_index(str(self.index_path))

    def add(self, embeddings: Iterable[np.ndarray], embedding_ids: Iterable[int]) -> None:
        vectors = np.vstack(list(embeddings)).astype("float32")
        ids = np.array(list(embedding_ids), dtype="int64")
        self.id_index.add_with_ids(vectors, ids)
        self.persist()

    def search(self, query_embedding: np.ndarray, k: int = 1):
        if self.id_index.ntotal == 0:
            return None

        q = query_embedding.astype("float32").reshape(1, self.dimension)
        distances, indices = self.id_index.search(q, k)
        idx = int(indices[0][0])
        if idx < 0:
            return None
        return idx, float(distances[0][0])

    def persist(self) -> None:
        faiss.write_index(self.id_index, str(self.index_path))
