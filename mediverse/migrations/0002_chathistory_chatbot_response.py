# Generated by Django 4.2.3 on 2024-03-22 11:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mediverse', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='chathistory',
            name='chatbot_response',
            field=models.TextField(default=''),
        ),
    ]
