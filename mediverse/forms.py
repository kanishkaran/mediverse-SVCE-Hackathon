from django import forms

class ChatbotForm(forms.Form):
    user_input = forms.CharField(max_length=200)
    widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Type your message here...'})