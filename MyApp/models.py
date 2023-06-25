# models.py

from django.db import models

class ImageUpload(models.Model):
    image = models.ImageField(upload_to='uploads/')
    classification = models.CharField(max_length=10)