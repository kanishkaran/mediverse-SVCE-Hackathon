from django.contrib.auth import login
from django.contrib.auth.forms import AuthenticationForm
from django.shortcuts import render
from .forms import CustomUserCreationForm
from django.http  import HttpResponseRedirect
from django.urls import reverse

def index(request):
    return render(request, "users/index.html")

def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return HttpResponseRedirect(reverse("mediverse:index"))
    else:
        form = AuthenticationForm()
    return render(request, 'users/login.html', {'form': form})

def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return HttpResponseRedirect(reverse("users:login"))
    else:
        form = CustomUserCreationForm()
    return render(request, 'users/register.html', {'form': form})