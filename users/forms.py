from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import CustomUser

class CustomUserCreationForm(UserCreationForm):
    class Meta(UserCreationForm.Meta):
        model = CustomUser
        fields = ('username', 'password1', 'password2', 'gender', 'email', 'dob', 'address', 'phone')
        
    dob = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))