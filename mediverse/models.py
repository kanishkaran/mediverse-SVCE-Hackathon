from django.db import models
from django.conf import settings
import json

class ChatHistory(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    user_input = models.TextField()
    chatbot_response_str = models.CharField(default='', max_length=255)  # For storing strings
    chatbot_response_list = models.TextField(default='')  # For storing lists
    timestamp = models.DateTimeField(auto_now_add=True)

    def save_response(self, response):
        if isinstance(response, list):
            self.chatbot_response_str = ''
            self.chatbot_response_list = json.dumps(response)
        elif isinstance(response, str):
            self.chatbot_response_str = response
            self.chatbot_response_list = ''
        else:
            raise ValueError("Invalid response type. Must be a string or a list.")

    def get_response(self):
        if self.chatbot_response_list:
            return json.loads(self.chatbot_response_list)
        else:
            return self.chatbot_response_str

    def __str__(self):
        return f'{self.user.username} - {self.user_input} - {self.get_response()} - {self.timestamp}'
    
class Medicine(models.Model):
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=255)
    uses = models.TextField()
    side_effects = models.TextField()
    price = models.DecimalField(max_digits=8, decimal_places=2)
    alternatives = models.CharField(max_length=255, null=True, blank=True)
    def __str__(self):
        return self.name

class Inventory(models.Model):
    medicine = models.OneToOneField(Medicine, primary_key=True, on_delete=models.CASCADE)
    price = models.DecimalField(max_digits=8, decimal_places=2)
    address = models.TextField()
    quantity = models.IntegerField()
    medicine = models.ForeignKey(Medicine, on_delete=models.CASCADE)
    def __str__(self):
        return self.medicine.name