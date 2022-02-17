from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [

        path('', views.index, name='pneumonia-home'),
        # path('login/', views.login, name='pneumonia-login'),
        path('register/', views.register, name="pneumonia-register"),
        path('upload/', views.upload_image_view, name="pneumonia-upload-image"),
        path('imageform/', views.image, name = "pneumonia-image"),
        path('history/', views.history, name="pneumonia-history"),
        path('action/',views.action, name="pneumonia-action")
]