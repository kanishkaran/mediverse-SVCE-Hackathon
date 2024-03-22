from django.contrib import admin
from .models import Medicine, Inventory
class InventoryAdmin(admin.ModelAdmin):
    filter_horizontal = ("medicine",)
admin.site.register(Medicine)
admin.site.register(Inventory)