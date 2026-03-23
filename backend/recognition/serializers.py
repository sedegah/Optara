from rest_framework import serializers

from recognition.models import RecognitionLog
from users.models import UserProfile


class RegisterSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=255)
    images = serializers.ListField(child=serializers.ImageField(), allow_empty=False)


class RecognizeSerializer(serializers.Serializer):
    image = serializers.ImageField()


class RecognitionLogSerializer(serializers.ModelSerializer):
    user_name = serializers.CharField(source="user.name", allow_null=True)

    class Meta:
        model = RecognitionLog
        fields = ["id", "timestamp", "user_name", "confidence", "label"]


class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = ["id", "name", "created_at"]
