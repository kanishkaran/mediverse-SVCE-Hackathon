import json
import pickle
import numpy as np
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from tensorflow.keras.models import load_model
import sqlite3
import google.generativeai as genai
from .medicine_extraction import extract_medicine_and_quantity
import markdown


genai.configure(api_key='GEMINI API KEY')

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('C:\programing\hackathon\git\medi\chatbot_resource\intents.json').read())

words = pickle.load(open('C:\programing\hackathon\git\medi\chatbot_resource\words.pkl', 'rb'))
classes = pickle.load(open('C:\programing\hackathon\git\medi\chatbot_resource\classes.pkl', 'rb'))

model = load_model('C:\programing\hackathon\git\medi\chatbot_resource\chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence = nltk.word_tokenize(sentence)
    lemmatized_words = [lemmatizer.lemmatize(i) for i in sentence]
    return lemmatized_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)

def markdown_to_html(markdown_text):
    html_text = markdown.markdown(markdown_text)
    return html_text

def get_gemini_response(question,prompt):
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content([prompt[0],question])
    return response.text

def get_gemini_response_common(message):
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content(message)
    return response.text

def read_sql_query(sql,db):
      conn=sqlite3.connect(db)
      cur=conn.cursor()
      cur.execute(sql)
      rows=cur.fetchall()
      conn.commit()
      conn.close()
      return rows


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in result:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


prompt=[
    """

"You're managing a pharmaceutical database named mediverse_medicine with the following columns - medicine_id, name, price, uses, side_effects, and alternatives. 
for example:
example 1: "Can you show me the details of a specific medicine?"
The SQL command will be something like this:
    SELECT * FROM mediverse_medicine WHERE name = 'medicine_name';

example 2: "What is the price of the medicine named 'Paracetamol'?"
    The SQL command will be something like this:
    SELECT price FROM mediverse_medicine WHERE name = 'Paracetamol';

example 3 - "What are the uses of 'Amoxicillin'?"
The SQL command will be something like this:
    SELECT uses FROM mediverse_medicine WHERE name = 'Amoxicillin';

example 4 - "Can you tell me the side effects of 'Lisinopril'?"
    The SQL command will be something like this:
    SELECT side_effects FROM mediverse_medicine WHERE name = 'Lisinopril';

example 5 - "Do you have any alternatives for 'Omeprazole'?"
    The SQL command will be something like this:
    SELECT alternatives FROM mediverse_medicine WHERE name = 'Omeprazole';

example 6 - "Is there a medicine available for headache?"
    The SQL command will be something like this:
    SELECT * FROM mediverse_medicine WHERE uses LIKE '%headache%';

example 7 -  "Show me all the medicines priced below 50₹."
    The SQL command will be something like this:
    SELECT * FROM mediverse_medicine WHERE price < 50;

example 8 - "Can you list all the medicines with 'tablet' in their name?"
    The SQL command will be something like this:
    SELECT * FROM mediverse_medicine WHERE name LIKE '%tablet%';

example 9 - "What are the side effects of the cheapest medicine available?"
    The SQL command will be something like this:
    SELECT side_effects FROM mediverse_medicine WHERE price = (SELECT MIN(price) FROM Medicines);

example 11 - "what are the medicines for fever?"
    The SQL command will be something like this:
    SELECT name from mediverse_medicine where uses LIKE '%fever%'
    if none give response as Not available
example 12 - "what are the medicines for drowsiness?"
    The SQL command will be something like this:
    SELECT name from mediverse_medicine where uses LIKE '%drowsiness%'
example 13 - "solutions for drowsiness?"
    The SQL command will be something like this:
    SELECT name from mediverse_medicine where uses LIKE '%drowsiness%'
example 14 - "solutions for fever?"
    The SQL command will be something like this:
    SELECT name from mediverse_medicine where uses LIKE '%fever%'

also the sql code should not have ``` in beginning or end and sql word in output
do not price the response


    You are also a medicine knowledge expert, if the question cannot be queried chat like a normal chatbot

    example 1 - "Hi "
    The text reply will be something like this:
    hello!
    example 2 - "Hello "
    The text reply will be something like this:
    What can I do for you Today?
    
    and try to avoid this error : sqlite3.OperationalError: near "I": syntax error
    """
]
alternative_prompt = [
    """

"You're managing a pharmaceutical database named mediverse_medicine with the following columns - medicine_id, name, price, uses, side_effects, and alternatives. 
for example:
example 1 - "Do you have any alternatives for 'Omeprazole'?"
    The SQL command will be something like this:
    SELECT alternatives FROM mediverse_medicine WHERE name = 'Omeprazole';
example 2 - "alternatives for 'Aspirin'?"
    The SQL command will be something like this:
    SELECT alternatives FROM mediverse_medicine WHERE name = 'Aspirin';
only the sql command should be your output
also the sql code should not have ``` in beginning or end and sql word in output
do not price the response

also try to avoid this error:sqlite3.OperationalError: near "*": syntax error"
    """
]

def get_response(message):
    intents_list = predict_class(message)
    # intents_json = intents
    tag = intents_list[0]['intent']
    # list_of_intents = intents_json['intents']
    # for i in list_of_intents:
    #     if i['tags'] == tag:
    #         result = random.choice(i['response'])
    #         break
    # return result
    if tag == "order":
        print("entered order context")
        med_name, quan = extract_medicine_and_quantity("message")
        return list(med_name)
        # if quan != None:
        #     return med_name, quan
        # else:
        #     return med_name
    elif tag == "alternative":
        response=get_gemini_response(message,alternative_prompt)
        # print(response)
        row= read_sql_query(response,"C:\programing\hackathon\git\medi\db.sqlite3")
        print("row:", row)
        response_list = []
    
        for tuple_item in row:
            # Extract the medicine names from the tuple
            medicine_names_str = tuple_item[0]
            
            # Split the medicine names string based on commas and strip any leading or trailing spaces
            medicine_names_list = [medicine.strip() for medicine in medicine_names_str.split(',')]
            
            # Loop through the extracted medicine names as strings
            for medicine_name in medicine_names_list:
                response_list.append(medicine_name)
        return response_list

    elif tag == "whatMed":
        response=get_gemini_response(message,prompt)
        # print(response)
        row= read_sql_query(response,"C:\programing\hackathon\git\medi\db.sqlite3")
        response_list = []
        for r in row:
            for i in r:
                response_list.append(i)
        return response_list
    else:
        return markdown_to_html(get_gemini_response_common(message))
if __name__ == "__main__":
    while True:
        message = input("")
        if message == 'quit':
            break
        # # extra = None
        print(get_response(message))
        # print(get_gemini_response_common(message))