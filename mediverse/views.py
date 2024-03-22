from django.shortcuts import render
from .chat import get_response
from .forms import ChatbotForm
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponseRedirect
from .cv import image_to_text
from .models import ChatHistory
from django.contrib.auth.decorators import login_required
from django.urls import reverse

@csrf_exempt
@login_required
def index(request):
    # username = request.user.username
    # gender = request.user.username

    chat_history = ChatHistory.objects.filter(user=request.user).order_by('-timestamp')[::-1]
    image_text = None
    
    chatbot_response = ''
    if request.method == 'POST':
        form = ChatbotForm(request.POST, request.FILES)
        if form.is_valid():
            user_input = form.cleaned_data['user_input']
            # image_file = request.FILES.get('image')

            # if image_file:
                # image_text = image_to_text(image_file)
                # chat_history.append(f"Image Text: {image_text}")
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
    medicine_detail = f"Details for {medicine_name}"
    return render(request, 'mediverse/medicine_detail.html', {'medicine_detail': medicine_detail})