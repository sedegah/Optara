from django.urls import path

from recognition.views import LogsView, RecognizeView, RegisterView

urlpatterns = [
    path("register/", RegisterView.as_view(), name="register"),
    path("recognize/", RecognizeView.as_view(), name="recognize"),
    path("logs/", LogsView.as_view(), name="logs"),
]
