import json

from django.db import models

from services.encryption import decrypt_vector, encrypt_vector
from users.models import UserProfile


class FaceEmbedding(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name="embeddings")
    encrypted_vector = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def set_vector(self, vector: list[float]) -> None:
        self.encrypted_vector = encrypt_vector(json.dumps(vector))

    def get_vector(self) -> list[float]:
        return json.loads(decrypt_vector(self.encrypted_vector))


class RecognitionLog(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(UserProfile, null=True, blank=True, on_delete=models.SET_NULL)
    confidence = models.FloatField()
    label = models.CharField(max_length=64)
