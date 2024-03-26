# Mediverse Chatbot for Ordering Medicine Online

Welcome to Mediverse, a chatbot designed to help you order medicine online conveniently.

## Getting Started

To run the program, follow these steps:

1. **Set Your API Key:**
   - Open [chatbot_resource/chat.py](https://github.com/tharun52/mediverse1/blob/bf4d3a1b751a91f9b7e8fe35314ce298767de846/chatbot_resource/chat.py) file.
   - Replace `GEMINI API KEY` with your own API key.

2. **Install Requirements:**
   - Open your terminal.
   - Run the command:
     ```
     pip install -r requirements.txt
     ```

3. **Run the Django Server:**
   - In the terminal, run:
     ```
     python manage.py runserver
     ```

4. **Optional: Manage Database as Admin**
   - To create a superuser for admin access, run:
     ```
     python manage.py createsuperuser
     ```
   - Login to the admin panel at [http://127.0.0.1:8000/admin/](http://127.0.0.1:8000/admin/) after running the server.

## Note
You may want to modify the inventory table in the admin panel since dummy data is used for example purposes.
