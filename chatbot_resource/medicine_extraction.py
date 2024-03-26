import re

def extract_medicine_and_quantity(input_text):
    print("text extraction....")
    # Define regular expressions for medicine names and quantities
    medicine_names = ["paracetamol",
    "Ibuprofen", "Aspirin", "Amoxicillin", "Omeprazole", "Lisinopril",
    "Atorvastatin", "Metformin", "Prednisone", "Albuterol", "Ciprofloxacin",
    "Diazepam", "Sertraline", "Citalopram", "Levothyroxine", "Losartan",
    "Warfarin", "Hydrochlorothiazide", "Pregabalin", "Tramadol", "Morphine",
    "Oxycodone", "Codeine", "Acetaminophen", "Naproxen", "Diphenhydramine",
    "Loratadine", "Dextromethorphan", "Budesonide", "Mometasone", "Fluticasone",
    "Montelukast", "Alprazolam", "Clonazepam", "Lorazepam", "Gabapentin",
    "Olanzapine", "Quetiapine", "Risperidone", "Clozapine", "Aripiprazole",
    "Venlafaxine", "Duloxetine", "Escitalopram", "Bupropion", "Mirtazapine",
    "Fluoxetine", "Sildenafil", "Tadalafil", "Vardenafil"
     ] 
    quantity_pattern = r'\b\d+\b'  # Matches one or more digits representing quantity
    
    # Initialize variables to store extracted information
    medicine_name = None
    quantity = None
    
    # Check for medicine names in the input text
    for name in medicine_names:
        if re.search(r'\b{}\b'.format(re.escape(name)), input_text, re.IGNORECASE):
            medicine_name = name
            break
    
    # Check for quantity in the input text
    quantity_match = re.search(quantity_pattern, input_text)
    if quantity_match:
        quantity = int(quantity_match.group())
    
    print(medicine_name)
    return medicine_name, quantity
