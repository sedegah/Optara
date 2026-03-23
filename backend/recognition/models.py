from django.db import models

from users.models import UserProfile


class FaceEmbedding(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name="embeddings")
    vector = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)


class RecognitionLog(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(UserProfile, null=True, blank=True, on_delete=models.SET_NULL)
    confidence = models.FloatField()
    label = models.CharField(max_length=64)
