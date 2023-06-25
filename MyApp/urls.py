# urls.py (MyApp)

from django.urls import path
from . import views

app_name = 'MyApp'

urlpatterns = [
    path('', views.home, name='home'),
    path('upload_image/', views.upload_image, name='upload_image'),
]