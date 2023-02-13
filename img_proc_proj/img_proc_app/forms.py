from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Image


class SignUpForm(UserCreationForm):
    class Meta:
        model = User
        fields = ('username', 'password1', 'password2', )


class SignInForm(forms.Form):
    class Meta:
        model = User
        fields = ('username', 'password', )


class ImageForm(forms.Form):
    class Meta:
        model = Image
        fields = ('user_id', 'path', )
