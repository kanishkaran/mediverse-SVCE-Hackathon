from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    gender_choices = (
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    )
    gender = models.CharField(max_length=1, choices=gender_choices)
    email = models.EmailField()
    dob = models.DateField()
    address = models.CharField(max_length=300)
    phone = models.PositiveIntegerField()
    def save(self, *args, **kwargs):
        if self.is_superuser:
            # Allow null for superusers
            self.gender = self.gender if self.gender else 'O'  # Default to 'Other' if gender is not specified
            self.dob = self.dob if self.dob else '2000-01-01'  # Default date of birth
            self.address = self.address if self.address else 'Unknown'  # Default address
            self.phone = self.phone if self.phone else '0000000000'  # Default phone number
        super().save(*args, **kwargs)
        
    def __str__(self):
        return self.username

