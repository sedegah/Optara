from django.urls import path

from recognition.views import RecognizeView, RegisterView, logs_view

urlpatterns = [
    path("register/", RegisterView.as_view(), name="register"),
    path("recognize/", RecognizeView.as_view(), name="recognize"),
    path("logs/", logs_view, name="logs"),
]
