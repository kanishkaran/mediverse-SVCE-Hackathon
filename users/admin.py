from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser

class CustomUserAdmin(UserAdmin):
    model = CustomUser
    fieldsets = UserAdmin.fieldsets + (
        (None, {'fields': ('gender', 'dob', 'address', 'phone')}),
    )
    add_fieldsets = UserAdmin.add_fieldsets + (
        (None, {'fields': ('gender', 'dob', 'address', 'phone')}),
    )
    list_display = ('username', 'email', 'first_name', 'last_name', 'gender', 'is_staff')
    search_fields = ('username', 'email', 'first_name', 'last_name')
    ordering = ('username',)

admin.site.register(CustomUser, CustomUserAdmin)
