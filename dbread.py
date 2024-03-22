import sqlite3
# helllo
# Connect to the SQLite database
conn = sqlite3.connect('db.sqlite3')

# Create a cursor object to execute SQL queries
cursor = conn.cursor()
medicines = medicines = [
    {"id": 2, "name": "Ibuprofen", "uses": "Fever, Pain relief, Inflammation", "side_effects": "Stomach upset, Headache", "price": 599.25, "alternatives": "Diclofenac, Naproxen, Aspirin"},
    {"id": 3, "name": "Aspirin", "uses": "Pain relief, Fever, Inflammation", "side_effects": "Stomach irritation, Bleeding", "price": 337.50, "alternatives": "Ibuprofen, Naproxen"},
    {"id": 4, "name": "Amoxicillin", "uses": "Bacterial infections", "side_effects": "Diarrhea, Allergic reaction", "price": 956.25, "alternatives": "Azithromycin, Cephalexin"},
    {"id": 5, "name": "Omeprazole", "uses": "Acid reflux, Ulcers", "side_effects": "Headache, Diarrhea", "price": 693.75, "alternatives": "Esomeprazole, Lansoprazole"},
    {"id": 6, "name": "Lisinopril", "uses": "High blood pressure, Heart failure", "side_effects": "Dizziness, Cough", "price": 1125.00, "alternatives": "Ramipril, Losartan"},
    {"id": 7, "name": "Atorvastatin", "uses": "High cholesterol", "side_effects": "Muscle pain, Liver problems", "price": 1387.50, "alternatives": "Simvastatin, Rosuvastatin"},
    {"id": 8, "name": "Metformin", "uses": "Type 2 diabetes", "side_effects": "Nausea, Diarrhea", "price": 525.00, "alternatives": "Glipizide, Sitagliptin"},
    {"id": 9, "name": "Prednisone", "uses": "Inflammation, Allergic reactions", "side_effects": "Weight gain, Mood changes", "price": 524.25, "alternatives": "Prednisolone, Dexamethasone"},
    {"id": 10, "name": "Albuterol", "uses": "Asthma, COPD", "side_effects": "Shakiness, Fast heartbeat", "price": 1687.50, "alternatives": "Levalbuterol, Ipratropium"},
    {"id": 11, "name": "Ciprofloxacin", "uses": "Bacterial infections", "side_effects": "Nausea, Diarrhea", "price": 824.25, "alternatives": "Levofloxacin, Cefuroxime"},
    {"id": 12, "name": "Diazepam", "uses": "Anxiety, Muscle spasms", "side_effects": "Drowsiness, Confusion", "price": 618.75, "alternatives": "Lorazepam, Clonazepam"},
    {"id": 13, "name": "Sertraline", "uses": "Depression, Anxiety", "side_effects": "Nausea, Insomnia", "price": 1012.50, "alternatives": "Fluoxetine, Escitalopram"},
    {"id": 14, "name": "Citalopram", "uses": "Depression", "side_effects": "Nausea, Dry mouth", "price": 881.25, "alternatives": "Escitalopram, Fluoxetine"},
    {"id": 15, "name": "Levothyroxine", "uses": "Hypothyroidism", "side_effects": "Weight loss, Insomnia", "price": 749.25, "alternatives": "Synthroid, Liothyronine"},
    {"id": 16, "name": "Losartan", "uses": "High blood pressure, Kidney problems", "side_effects": "Dizziness, Fatigue", "price": 1068.75, "alternatives": "Valsartan, Olmesartan"},
    {"id": 17, "name": "Warfarin", "uses": "Blood clots", "side_effects": "Bleeding, Bruising", "price": 487.50, "alternatives": "Dabigatran, Rivaroxaban"},
    {"id": 18, "name": "Hydrochlorothiazide", "uses": "High blood pressure", "side_effects": "Dizziness, Muscle cramps", "price": 431.25, "alternatives": "Chlorthalidone, Indapamide"},
    {"id": 19, "name": "Pregabalin", "uses": "Neuropathic pain, Anxiety", "side_effects": "Dizziness, Drowsiness", "price": 1312.50, "alternatives": "Gabapentin, Duloxetine"},
    {"id": 20, "name": "Tramadol", "uses": "Moderate to severe pain", "side_effects": "Nausea, Dizziness", "price": 749.25, "alternatives": "Codeine, Oxycodone"},
    {"id": 21, "name": "Morphine", "uses": "Severe pain", "side_effects": "Drowsiness, Constipation", "price": 1949.25, "alternatives": "Oxycodone, Hydromorphone"},
    {"id": 22, "name": "Oxycodone", "uses": "Moderate to severe pain", "side_effects": "Nausea, Dizziness", "price": 974.25, "alternatives": "Morphine, Hydrocodone"},
    {"id": 23, "name": "Codeine", "uses": "Mild to moderate pain, Cough", "side_effects": "Nausea, Constipation", "price": 637.50, "alternatives": "Tramadol, Hydrocodone"},
    {"id": 24, "name": "Acetaminophen", "uses": "Pain relief, Fever", "side_effects": "Rarely, allergic reactions", "price": 374.25, "alternatives": ""},
    {"id": 25, "name": "Naproxen", "uses": "Pain relief, Inflammation", "side_effects": "Stomach upset, Headache", "price": 524.25, "alternatives": "Ibuprofen, Diclofenac"},
    {"id": 26, "name": "Diclofenac", "uses": "Pain relief, Inflammation", "side_effects": "Stomach upset, Headache", "price": 599.25, "alternatives": "Ibuprofen, Naproxen"},
    {"id": 27, "name": "Azithromycin", "uses": "Bacterial infections", "side_effects": "Diarrhea, Stomach pain", "price": 1124.25, "alternatives": "Amoxicillin, Cephalexin"},
    {"id": 28, "name": "Cephalexin", "uses": "Bacterial infections", "side_effects": "Diarrhea, Stomach pain", "price": 799.25, "alternatives": "Amoxicillin, Azithromycin"},
    {"id": 29, "name": "Esomeprazole", "uses": "Acid reflux, Ulcers", "side_effects": "Headache, Diarrhea", "price": 1068.75, "alternatives": "Omeprazole, Lansoprazole"},
    {"id": 30, "name": "Lansoprazole", "uses": "Acid reflux, Ulcers", "side_effects": "Headache, Diarrhea", "price": 1187.50, "alternatives": "Omeprazole, Esomeprazole"},
    {"id": 31, "name": "Ramipril", "uses": "High blood pressure, Heart failure", "side_effects": "Dizziness, Cough", "price": 856.25, "alternatives": "Lisinopril, Losartan"},
    {"id": 32, "name": "Simvastatin", "uses": "High cholesterol", "side_effects": "Muscle pain, Liver problems", "price": 1187.50, "alternatives": "Atorvastatin, Rosuvastatin"},
    {"id": 33, "name": "Rosuvastatin", "uses": "High cholesterol", "side_effects": "Muscle pain, Liver problems", "price": 1312.50, "alternatives": "Atorvastatin, Simvastatin"},
    {"id": 34, "name": "Glipizide", "uses": "Type 2 diabetes", "side_effects": "Nausea, Diarrhea", "price": 462.50, "alternatives": "Metformin, Sitagliptin"},
    {"id": 35, "name": "Sitagliptin", "uses": "Type 2 diabetes", "side_effects": "Hypoglycemia, Stomach pain", "price": 824.25, "alternatives": "Metformin, Glipizide"},
    {"id": 36, "name": "Prednisolone", "uses": "Inflammation, Allergic reactions", "side_effects": "Weight gain, Mood changes", "price": 587.50, "alternatives": "Prednisone, Dexamethasone"},
    {"id": 37, "name": "Dexamethasone", "uses": "Inflammation, Allergic reactions", "side_effects": "Weight gain, Mood changes", "price": 562.50, "alternatives": "Prednisone, Prednisolone"},
    {"id": 38, "name": "Levalbuterol", "uses": "Asthma, COPD", "side_effects": "Shakiness, Fast heartbeat", "price": 1825.00, "alternatives": "Albuterol, Ipratropium"},
    {"id": 39, "name": "Ipratropium", "uses": "Asthma, COPD", "side_effects": "Dry mouth, Blurred vision", "price": 1187.50, "alternatives": "Albuterol, Levalbuterol"},
    {"id": 40, "name": "Levofloxacin", "uses": "Bacterial infections", "side_effects": "Nausea, Diarrhea", "price": 887.50, "alternatives": "Ciprofloxacin, Cefuroxime"},
    {"id": 41, "name": "Cefuroxime", "uses": "Bacterial infections", "side_effects": "Nausea, Diarrhea", "price": 756.25, "alternatives": "Ciprofloxacin, Levofloxacin"},
    {"id": 42, "name": "Lorazepam", "uses": "Anxiety, Muscle spasms", "side_effects": "Drowsiness, Confusion", "price": 674.25, "alternatives": "Diazepam, Clonazepam"},
    {"id": 43, "name": "Clonazepam", "uses": "Anxiety, Seizure disorders", "side_effects": "Drowsiness, Fatigue", "price": 499.25, "alternatives": "Diazepam, Lorazepam"},
    {"id": 44, "name": "Fluoxetine", "uses": "Depression, Anxiety, OCD", "side_effects": "Nausea, Insomnia", "price": 437.50, "alternatives": "Sertraline, Escitalopram"},
    {"id": 45, "name": "Escitalopram", "uses": "Depression, Anxiety", "side_effects": "Nausea, Insomnia", "price": 474.25, "alternatives": "Sertraline, Fluoxetine"},
    {"id": 46, "name": "Synthroid", "uses": "Hypothyroidism", "side_effects": "Weight loss, Insomnia", "price": 624.25, "alternatives": "Levothyroxine, Liothyronine"},
    {"id": 47, "name": "Liothyronine", "uses": "Hypothyroidism", "side_effects": "Weight loss, Insomnia", "price": 899.25, "alternatives": "Levothyroxine, Synthroid"},
    {"id": 48, "name": "Valsartan", "uses": "High blood pressure, Heart failure", "side_effects": "Dizziness, Fatigue", "price": 1124.25, "alternatives": "Losartan, Olmesartan"},
    {"id": 49, "name": "Olmesartan", "uses": "High blood pressure, Heart failure", "side_effects": "Dizziness, Fatigue", "price": 1187.50, "alternatives": "Losartan"}]
# Define the SQL INSERT statement
sql_insert = "INSERT INTO mediverse_medicine (id, name, uses, side_effects, price, alternatives) VALUES (?, ?, ?, ?, ?, ?)"

# Insert data into the table
for medicine in medicines:
    values = (medicine['id'], medicine['name'], medicine['uses'], medicine['side_effects'], medicine['price'], medicine['alternatives'])
    cursor.execute(sql_insert, values)

# Commit the changes to the database
conn.commit()

# Close the cursor and connection
cursor.close()
conn.close()
