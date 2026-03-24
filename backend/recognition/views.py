from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.db import transaction
from django.http import JsonResponse
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from recognition.models import FaceEmbedding, RecognitionLog
from recognition.serializers import RecognizeSerializer, RegisterSerializer
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
            try:
                vector = extract_embedding_from_upload(image)
            except ValueError:
                continue

            embedding = FaceEmbedding(user=user)
            embedding.set_vector(vector.tolist())
            embedding.save()
            embeddings.append(vector)
            embedding_ids.append(embedding.id)

        if not embeddings:
            user.delete()
            return Response({"error": "No valid faces detected in any uploaded images by backend."}, status=status.HTTP_400_BAD_REQUEST)

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
        return Response(
            {"label": "MATCH", "user_id": embedding.user.id, "name": embedding.user.name, "confidence": log.confidence}
        )


def logs_view(request):
    logs = RecognitionLog.objects.select_related("user").order_by("-timestamp")[:100]
    payload = [
        {
            "id": item.id,
            "timestamp": item.timestamp.isoformat(),
            "user_name": item.user.name if item.user else None,
            "confidence": item.confidence,
            "label": item.label,
        }
        for item in logs
    ]
    return JsonResponse(payload, safe=False)
