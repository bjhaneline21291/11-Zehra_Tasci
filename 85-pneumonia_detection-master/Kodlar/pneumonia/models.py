from django.db import models

# Create your models here.
class xray_image(models.Model):
    photo =models.FileField(upload_to="xray_photo/")
    uploaded_by = models.CharField(max_length=50, default="Guest")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.uploaded_at, self.photo, self.uploaded_by