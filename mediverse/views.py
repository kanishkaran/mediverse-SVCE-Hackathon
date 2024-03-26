from django.shortcuts import render
from .forms import ChatbotForm
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponseRedirect
from .models import ChatHistory
from django.contrib.auth.decorators import login_required
from django.urls import reverse
from chatbot_resource.chat import get_response
from mediverse.models import Medicine, Inventory
from django.contrib import messages

def get_medicine_details(medicine_name):
    try:
        # Query the Medicine model to get the medicine details
        medicine = Medicine.objects.get(name=medicine_name)
        
        # Query the Inventory model to get the price and quantity details
        inventory = Inventory.objects.get(medicine=medicine)
        print("medicine id :",medicine.id)
        # Build the medicine details string using f-strings
        medicine_detail = {
            'id': medicine.id,
            'name': medicine.name,
            'price': str(inventory.price),
            'uses': medicine.uses,
            'side_effects': medicine.side_effects,
            'quantity': str(inventory.quantity)
        }
        return medicine_detail
    except Medicine.DoesNotExist:
        return f"Medicine '{medicine_name}' not found in the database."
    except Inventory.DoesNotExist:
        return f"Inventory details not found for medicine '{medicine_name}'."
    
@csrf_exempt
@login_required
def index(request):
    chat_history = ChatHistory.objects.filter(user=request.user).order_by('-timestamp')[::-1]
    
    chatbot_response = ''
    if request.method == 'POST':
        form = ChatbotForm(request.POST, request.FILES)
        if form.is_valid():
            user_input = form.cleaned_data['user_input']
            chatbot_response = get_response(user_input)
            if isinstance(chatbot_response, list):
                ChatHistory.objects.create(
                    user=request.user, 
                    user_input=user_input, 
                    chatbot_response_list=chatbot_response
                )
            if isinstance(chatbot_response, str):
                ChatHistory.objects.create(
                    user=request.user, 
                    user_input=user_input, 
                    chatbot_response_str=chatbot_response
                )
            
            return HttpResponseRedirect(request.path_info)
    else:
        form = ChatbotForm()
    return render(request, 'mediverse/chatbot.html', {
        'form': form, 
        'chat_history': chat_history, 
        'chatbot_response':chatbot_response
    })

def clear_chat_history(request):
    if request.method == 'POST':
        clear_confirmation = request.POST.get('clear_confirmation')
        if clear_confirmation == '1':
            # Clear chat history for the current user
            ChatHistory.objects.filter(user=request.user).delete()
    return HttpResponseRedirect(reverse('mediverse:index'))

def medicine_detail(request, medicine_name):
    # Retrieve details of the medicine based on the name
    # Example: medicine_detail = get_medicine_detail(medicine_name)
    medicine_detail = get_medicine_details(medicine_name)                                      
    return render(request, 'mediverse/medicine_detail.html', {'medicine_detail': medicine_detail})

def payment_view(request):
    if request.method == 'POST':
        medicine_id = request.POST.get('medicine_id')
        quantity = int(request.POST.get('quantity'))
        try:
            inventory = Inventory.objects.get(medicine_id=medicine_id)
            if inventory.quantity >= quantity:
                inventory.quantity -= quantity
                inventory.save()
                messages.success(request, 'Payment successful!')
                return HttpResponseRedirect(reverse('mediverse:payment'))
            else:
                messages.error(request, 'Insufficient quantity in inventory.')
        except Inventory.DoesNotExist:
            messages.error(request, 'Inventory details not found.')
    return HttpResponseRedirect(reverse('mediverse:confirm_payment')) 

def confirm_payment(request):
    if request.method == 'POST' and 'payment_confirmation' in request.POST:
        return HttpResponseRedirect(reverse('mediverse:index'))
    else:
        return render(request, 'mediverse/payment.html')