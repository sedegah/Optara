from __future__ import annotations

import io

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image

mtcnn = MTCNN(image_size=160, margin=20)
model = InceptionResnetV1(pretrained="vggface2").eval()


def extract_embedding_from_upload(uploaded_file) -> np.ndarray:
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    face_tensor = mtcnn(image)
    if face_tensor is None:
        raise ValueError("No face detected in uploaded image.")

    with torch.no_grad():
        embedding = model(face_tensor.unsqueeze(0)).squeeze(0)

    return embedding.detach().cpu().numpy().astype("float32")
