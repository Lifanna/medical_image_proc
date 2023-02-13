from django.db import models
from django.contrib.auth.models import User


"""User class"""
class UserAdditional(User):
    firstname = models.CharField("FirstName", max_length=255, null=True)

    lastname = models.CharField("LastName", max_length=255, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    
    updated_at = models.DateTimeField(auto_now=True)


"""Result class"""
class Result(models.Model):
    title = models.CharField("FirstName", max_length=255, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    
    updated_at = models.DateTimeField(auto_now=True)


"""Image class with results of checking"""
class Image(models.Model):
    path = models.CharField("path", max_length=255, null=True)

    checked = models.BooleanField("Status of checking", default=False)

    user = models.ForeignKey(User, null=True, on_delete=models.SET_NULL)
    
    result = models.ForeignKey(Result, null=True, on_delete=models.SET_NULL)

    created_at = models.DateTimeField(auto_now_add=True)

    updated_at = models.DateTimeField(auto_now=True)

    image_file = models.ImageField(upload_to='images/%d_%m_%Y_%H_%M_%S/', blank=True, null=True)
