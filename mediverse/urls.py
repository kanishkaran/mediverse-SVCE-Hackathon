from django.urls import path
from . import views

app_name = "mediverse"
urlpatterns = [
    path("", views.index, name="index"),
    path('clear/', views.clear_chat_history, name='clear_chat_history'),
    path('medicine-detail/<str:medicine_name>/', views.medicine_detail, name='medicine_detail'),
]