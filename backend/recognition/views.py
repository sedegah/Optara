from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.db import transaction
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from recognition.models import FaceEmbedding, RecognitionLog
from recognition.serializers import RecognitionLogSerializer, RecognizeSerializer, RegisterSerializer
from services.embeddings import extract_embedding_from_upload
from services.faiss_index import FaissEngine
from users.models import UserProfile

INDEX_PATH = Path(settings.BASE_DIR).parent / "storage" / "faiss.index"
faiss_engine = FaissEngine(index_path=INDEX_PATH)


class RegisterView(APIView):
    @transaction.atomic
    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user = UserProfile.objects.create(name=serializer.validated_data["name"])
        embeddings = []
        embedding_ids = []

        for image in serializer.validated_data["images"]:
            vector = extract_embedding_from_upload(image)
            embedding = FaceEmbedding.objects.create(user=user, vector=vector.tolist())
            embeddings.append(vector)
            embedding_ids.append(embedding.id)

        faiss_engine.add(embeddings, embedding_ids)
        return Response({"user_id": user.id, "embedding_count": len(embeddings)}, status=status.HTTP_201_CREATED)


class RecognizeView(APIView):
    def post(self, request):
        serializer = RecognizeSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        query_vector = extract_embedding_from_upload(serializer.validated_data["image"])
        result = faiss_engine.search(query_vector)

        if result is None:
            log = RecognitionLog.objects.create(user=None, confidence=0.0, label="UNKNOWN")
            return Response({"label": log.label, "confidence": log.confidence})

        embedding_id, distance = result
        confidence = float(1.0 / (1.0 + distance))

        threshold = 0.60
        if confidence < threshold:
            log = RecognitionLog.objects.create(user=None, confidence=confidence, label="UNKNOWN")
            return Response({"label": log.label, "confidence": confidence})

        embedding = FaceEmbedding.objects.select_related("user").get(id=embedding_id)
        log = RecognitionLog.objects.create(user=embedding.user, confidence=confidence, label="MATCH")
        return Response({"label": "MATCH", "user_id": embedding.user.id, "name": embedding.user.name, "confidence": log.confidence})


class LogsView(APIView):
    def get(self, request):
        logs = RecognitionLog.objects.select_related("user").order_by("-timestamp")[:100]
        return Response(RecognitionLogSerializer(logs, many=True).data)
