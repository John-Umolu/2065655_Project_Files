# Python program to translate
# speech to text and text to speech
import speech_recognition as sr
import pyttsx3
import requests
import json
from datetime import datetime
import pandas as pd
import numpy as np
import tkinter
from tkinter import *
from PIL import Image, ImageOps

from tkinter.filedialog import askopenfilename

import os
import sys
# Libraries for heart disease prediction
import pickle
from pathlib import Path
# play audio file
from pygame import mixer

from keras.models import load_model

# Initializing pygame mixer
mixer.init()

# Loading Music File
mixer.music.load('jingle.mp3')

# Playing Music with Pygame
mixer.music.play()

# Heart Disease Model
# For Knn Model: hdp_knn.pkl
# For SVM Model: hdp_svm.sav
hdp_model = Path('hdp_knn.pkl')

# Create a new Tkinter Form for NI Reg
master = tkinter.Tk()
master1 = tkinter.Tk()

# Create a text box to enter NI number
e = Entry(master, width=30)
master.withdraw()
master1.withdraw()

# create tkinter form for image upload prediction
first = tkinter.Tk()
second = tkinter.Tk()

# quite tkinter form to avoid duplicate
first.withdraw()
second.withdraw()

# create tkinter form for order food list
first1 = tkinter.Tk()
second1 = tkinter.Tk()

# creating drop down list
variable_first1 = StringVar(first1)
variable_second1 = StringVar(second1)

# quite tkinter form to avoid duplicate
first1.withdraw()
second1.withdraw()

# Load the model
model_food = load_model('keras_model.h5')

# Read Food Dataset From Webserver
food_data = pd.read_csv('food_data.csv')

# Change food category names to upper case
food_data['Category'] = food_data['Category'].str.upper()

# Read Heart Disease Dataset From Webserver
heart_data = pd.read_csv('heart.csv')

# Assign Integer Values To Some Columns With String Names
heart_data.loc[heart_data['Sex'] == 'M', 'Sex'] = 1
heart_data.loc[heart_data['Sex'] == 'F', 'Sex'] = 0
heart_data.loc[heart_data['ChestPainType'] == 'TA', 'ChestPainType'] = 0
heart_data.loc[heart_data['ChestPainType'] == 'ATA', 'ChestPainType'] = 1
heart_data.loc[heart_data['ChestPainType'] == 'NAP', 'ChestPainType'] = 2
heart_data.loc[heart_data['ChestPainType'] == 'ASY', 'ChestPainType'] = 3
heart_data.loc[heart_data['RestingECG'] == 'Normal', 'RestingECG'] = 1
heart_data.loc[heart_data['RestingECG'] == 'ST', 'RestingECG'] = 2
heart_data.loc[heart_data['RestingECG'] == 'LVH', 'RestingECG'] = 3
heart_data.loc[heart_data['ExerciseAngina'] == 'N', 'ExerciseAngina'] = 0
heart_data.loc[heart_data['ExerciseAngina'] == 'Y', 'ExerciseAngina'] = 1
heart_data.loc[heart_data['ST_Slope'] == 'Up', 'ST_Slope'] = 1
heart_data.loc[heart_data['ST_Slope'] == 'Flat', 'ST_Slope'] = 2
heart_data.loc[heart_data['ST_Slope'] == 'Down', 'ST_Slope'] = 3

# Declared Variables To Be Used
order_confirmation: int = 0
prev_command: str = ""
last_requested: str = ""
process: int = 0
errorState: int = 0
prevQuestion: str = ""
noResponse: int = 0
responseFound: int = 0
serverResponse: str = ""
newQuestion: str = ""
confirmAnswer: int = 0
recogSpeech: str = ""
newResponse: str = ""
cityResponse: str = ""
ni_number: str = ""
finalOrderCheck: int = 0
food_descriptions = []
Food_Category: str = ""
Food_Description: str = ""
confirm_suggestion: int = 0
suggested_food: str = ""
image_prediction: int = 0
new_preRequested: str = ""
Order_Time: str = ""
Order_Date: str = ""
Order_Frequency: str = ""
heart_condition: int = 0
count: int = 0
count1: int = 0
dont_repeat: int = 0
dont_repeat1: int = 0
entered_ni: str = ""
first_start: int = 0
exit_count: int = 0
start_variable: int = 0

# Initialize the recognizer
r = sr.Recognizer()

# Checks if User NI number is saved
if ni_number == "" or len(ni_number) == 0:
    my_file = Path('NI_Data.txt')
    if my_file.exists():
        # Kill NI Form
        master1.destroy()
        # read labels text file into list
        read_labels = open('NI_Data.txt', 'r')
        ni_number = read_labels.readline()


# Function to convert text to
# speech
def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    rate = engine.getProperty('rate')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 145)
    engine.say(command)
    engine.runAndWait()


# Function to save new User NI Number
def saveNI():
    global entered_ni
    global ni_number
    # Gets the new value in the tkinter text box
    with open("NI_Data.txt", "w") as text_file:
        text_file.write(entered_ni)
        ni_number = entered_ni


# Close NI Tkinter form
def close_niform():
    global entered_ni
    global master
    global master1
    entered_ni = str(e.get())
    # Close tkinter dialogue box
    master1.quit()
    master1.destroy()
    saveNI()


# Function to display NI dialogue box prompt for user
def requestNI():
    global master1
    global e
    try:
        # Kill previous form
        master.quit()
        master.destroy()

        # Create a new Tkinter Form
        master1 = tkinter.Tk()

        # Create a text box to enter NI number
        e = Entry(master1, width=30)

        # Tkinter dialogue settings
        master1.title('Enter Your NI Number Below')
        master1.geometry("300x60")

        # Tkinter textbox settings
        # Create a text box to enter NI number
        e = Entry(master1, width=30)
        e.pack()
        e.focus_set()

        # Tkinter button settings
        b = Button(master1, text="Click To Save Data", width=20, command=close_niform)
        b.pack()

        print("PLEASE ENTER YOUR NATIONAL INSURANCE NUMBER")
        SpeakText("PLEASE ENTER YOUR NATIONAL INSURANCE NUMBER")

        # Start the program loop
        master1.mainloop()

    except Exception as e2:
        print("ERROR (NI ENTERING FORM): ", e2)


# Function to save Heart Disease Prediction to online database
def saveNewPrediction(prediction):
    try:
        response = requests.get("https://akifagoelectronics.com/UoW_2065655_Project/saveData.php?User_NI_Number="
                                + ni_number + "&Pred_Value=" + prediction, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
        })
        # Check for http OK response code
        if response.status_code == 200:
            # Do this when new data update is successful
            print("NEW HEART DISEASE PREDICTION UPLOADED TO ONLINE DATABASE SUCCESSFULLY.")
        else:
            print('Not Connected!')
    except requests.exceptions.RequestException as e:
        print(e)


# Function to check question and retrieve a response saved on the database
def predictHeartDisease():
    global heart_condition
    global ni_number
    # fullpath = os.path.join(location, 'hdp_model.pkl')
    heart_model = pickle.load(open(hdp_model, 'rb'))
    # heart_model = joblib.load(hdp_model)
    # place a call on the webserver to get user hospital data
    if not ni_number == "" or len(ni_number) > 0:
        try:
            response = requests.get("https://akifagoelectronics.com/UoW_2065655_Project/5@VIoTphp2Files/getRecord.php?NI_Number=" + ni_number, headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
            })
            # Check for http OK response code
            if response.status_code == 200:
                # Get JSON Objects form response text
                user = json.loads(response.text)
                Age = int(user[0]['Age'])
                Sex = int(user[0]['Sex'])
                ChestPainType = int(user[0]['ChestPainType'])
                RestingBP = int(user[0]['RestingBP'])
                Cholesterol = int(user[0]['Cholesterol'])
                FastingBS = int(user[0]['FastingBS'])
                RestingECG = int(user[0]['RestingECG'])
                MaxHR = int(user[0]['MaxHR'])
                ExerciseAngina = int(user[0]['ExerciseAngina'])
                Oldpeak = float(user[0]['Oldpeak'])
                ST_Slope = int(user[0]['ST_Slope'])
                HeartDisease = int(user[0]['HeartDisease'])

                # convert json response to np array to predict
                values = np.array([[Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope]])
                print('USER HOSPITAL MEDICAL DATA: ',  values)

                # predict heart disease using loaded model
                prediction = heart_model.predict(values)
                prediction = int(prediction[0])
                heart_condition = prediction

                # print heart condition
                print("PREVIOUS HEART DISEASE PREDICTION ", HeartDisease)
                print("CLASSIFIER HEART DISEASE PREDICTION: ", prediction)

                # Stop music
                mixer.music.stop()

                # Do this when new data update is successful
                if prediction == 0:
                    print("CONGRATULATION! YOU DONT HAVE HEART DISEASE. STAY HEALTHY")
                    SpeakText("CONGRATULATION! YOU DONT HAVE HEART DISEASE. STAY HEALTHY")

                elif prediction == 1:
                    print("OOPS! YOU HAVE CHANCES OF HEART DISEASE. PLEASE CONTACT THE NEAREST HOSPITAL")
                    SpeakText("OOPS! YOU HAVE CHANCES OF HEART DISEASE. PLEASE CONTACT THE NEAREST HOSPITAL")

                # Upload new prediction to webserver
                saveNewPrediction(str(prediction))

            else:
                print('Not Connected!')
        except requests.exceptions.RequestException as e1:
            print(e1)


# Check if trained model file exists
if hdp_model.exists():
    # call the predict function
    predictHeartDisease()
# train the model if it does not exist
else:
    import train_models


# Function to check question and retrieve a response saved on the database
def getResponse(value):
    global responseFound
    global serverResponse
    global newQuestion
    global noResponse
    global recogSpeech

    try:
        response = requests.get("https://www.akifagoelectronics.com/robot@AE2527/robotRead.php", headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
        })
        # Check for http OK response code
        if response.status_code == 200:
            # Get JSON Objects form response text
            users = json.loads(response.text)
            # Loop through all received JSON Objects
            for user in users:
                # Get asked question into a string and convert it to upper case
                newQuestion = str(value).upper()
                # Get saved server questions and convert it to upper case
                serverQuestion: str = str(user['question']).upper()
                # Confirm if asked question is equal to save server question
                if serverQuestion == newQuestion:
                    responseFound = 1
                    serverResponse = user['response']
            if responseFound == 1:
                responseFound = 0
                # Output the saved response from the server
                SpeakText(serverResponse)
            else:
                noResponse = 1
                recogSpeech = newQuestion
                SpeakText("I don't have the right response now, please what should i say next time when " + newQuestion
                          + " is asked.")
        else:
            print('Not Connected!')
    except requests.exceptions.RequestException as e1:
        print(e1)


# Function to save ordered food items on online database
def saveNewOrder(food_category, food_description):
    global recogSpeech
    try:
        response = requests.get("https://akifagoelectronics.com/UoW_2065655_Project/saveData.php?Food_NI_Number=" + ni_number
                                + "&Food_Category=" + food_category + "&Food_Description=" + food_description, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
        })
        # Check for http OK response code
        if response.status_code == 200:
            # Get JSON Objects form response text
            print("YOUR PURCHASE OF " + food_description.upper() + " WAS SUCCESSFUL. THANK YOU.")
            SpeakText("YOUR PURCHASE OF " + food_description + " WAS SUCCESSFUL. THANK YOU.")
        else:
            print('Not Connected!')
    except requests.exceptions.RequestException as e:
        print(e)


# Function to retrieve user cholesterol level saved on the hospital webpage form
def getCholesterolLevel():
    global finalOrderCheck
    global order_confirmation
    global confirm_suggestion
    global suggested_food
    try:
        response = requests.get("https://akifagoelectronics.com/UoW_2065655_Project/5@VIoTphp2Files/checkCholesterol.php?NI_Number=" + ni_number, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
        })
        # Check for http OK response code
        if response.status_code == 200:
            # Get JSON Objects form response text
            user = json.loads(response.text)
            cholesterol_level: int = 0
            food_allergies: str = ""
            # Loop through all received JSON Objects
            for obj in user:
                # Get cholesterol level
                cholesterol_level = int(obj['Cholesterol'])
                # get food allergies
                food_allergies = str(obj['Food_Allergies'])
            # Print values
            print("YOUR CHOLESTEROL LEVEL: ", cholesterol_level, " || FOOD ALLERGIES: ", food_allergies.upper())
            finalOrderCheck = 0
            # Normal Cholesterol Level Is < 200
            if cholesterol_level >= 200 or heart_condition == 1:
                try:
                    # Check for command keywords
                    if food_data['Category'].str.contains(last_requested).any():
                        # get food description values using the requested name
                        description_rows = food_data.loc[(food_data['Category'] == last_requested),
                                                         'Description']
                        # get food cholesterol values using the requested name
                        cholesterol_rows = food_data.loc[(food_data['Category'] == last_requested),
                                                         'Data.Cholesterol']
                        # get the least cholesterol index from the selected dataframe
                        cholesterol_array = cholesterol_rows.to_numpy()
                        least_cholesterol_index = np.argmin(cholesterol_array)
                        # get the least cholesterol value from the selected dataframe
                        least_cholesterol = cholesterol_array[least_cholesterol_index]
                        # get the nth item from the dataframe
                        suggested_food = ' '.join(description_rows.take([least_cholesterol_index]))
                        # Search for food allergies in suggested food
                        new_food_allergies = food_allergies.upper()
                        allergy_state: int = 0
                        # Check for empty space and remove them from string
                        if ' ' in new_food_allergies:
                            new_food_allergies = new_food_allergies.replace(' ', '')
                        # if allergy is multiple food and check for ',' and split string
                        if ',' in new_food_allergies:
                            split_allergies = new_food_allergies.split(',')
                            for a in range(len(split_allergies)):
                                substring = split_allergies[a]
                                if suggested_food.upper().find(substring.upper()) != -1:
                                    # Found allergy food in suggested food
                                    allergy_state = 1
                                else:
                                    # Did not find allergy in suggested food
                                    allergy_state = 0
                        # if allergy is a single food
                        if ',' not in new_food_allergies:
                            if suggested_food.upper().find(new_food_allergies) != -1:
                                # Found allergy food in suggested food
                                allergy_state = 1
                            else:
                                # Did not find allergy in suggested food
                                allergy_state = 0
                        # if allergies is not found in suggested food
                        if len(suggested_food) > 0 and allergy_state == 0:
                            print("SUGGESTED FOOD: ", str(suggested_food).upper(),
                                  " || FOOD CHOLESTEROL CONTENT: ", least_cholesterol)
                            # Convert text to speech
                            SpeakText("ACCORDING TO OUR DATABASE, YOU ARE ALLERGIC TO " + food_allergies.upper()
                                      + ". MAY I SUGGEST THAT THIS TIME YOU PURCHASE " + str(suggested_food)
                                      + " BECAUSE YOUR CHOLESTEROL LEVEL IS " + str(cholesterol_level)
                                      + " WHICH IS NOT GOOD. SHOULD I PROCEED WITH THE ORDER?"
                                      + " PLEASE RESPOND WITH A YES OR A NO.")
                            order_confirmation = 0
                            finalOrderCheck = 0
                            confirm_suggestion = 1
                        # if allergies is found in suggested food
                        elif len(suggested_food) > 0 and allergy_state == 1:
                            print("SUGGESTED FOOD: ", str(suggested_food).upper(),
                                  " || FOOD CHOLESTEROL CONTENT: ", least_cholesterol)
                            # Convert text to speech
                            SpeakText("ACCORDING TO OUR DATABASE, YOU ARE ALLERGIC TO " + food_allergies.upper()
                                      + ". THE SUGGESTED FOOD WHICH IS " + str(suggested_food)
                                      + " CANNOT BE PURCHASED NOW BECAUSE IT HAS SOME FOOD CONTENTS THAT YOU ARE"
                                      + " ALLERGIC TO. YOU ARE ADVISED TO VISIT A DOCTOR FOR ADVICE ON THE RIGHT FOOD"
                                      + " TO EAT BECAUSE YOUR CHOLESTEROL LEVEL IS " + str(cholesterol_level)
                                      + " WHICH IS NOT GOOD. THANK YOU. HAVE A NICE DAY. STAY HEALTHY.")
                            order_confirmation = 0
                            finalOrderCheck = 0
                            confirm_suggestion = 0
                        else:
                            print("SELECTED DATA ROW IS EMPTY")
                    else:
                        print("SORRY", last_requested, "FOOD ITEM NOT FOUND ON THE LIST")

                except ValueError as e2:
                    if order_confirmation == 1:
                        SpeakText("PLEASE I AM WAITING FOR A YES OR NO RESPONSE FROM YOU TO THE QUESTION: DID YOU SAY "
                                  + str(prev_command))
                    print("PLEASE I AM WAITING FOR A YES OR NO RESPONSE FROM YOU TO THE QUESTION: DID YOU SAY "
                          + str(prev_command).upper())
            else:
                saveNewOrder(Food_Category, Food_Description)
        else:
            print('NOT CONNECTED!')
    except requests.exceptions.RequestException as e1:
        print(e1)


# Function to save the selected food description of the user on the online database
def proceedOrder():
    global variable_first1
    global variable_second1
    global count1

    if count1 == 1:
        selected_list = str(variable_first1.get())
        saveNewOrder(last_requested, selected_list)
        # Close tkinter dialogue box
        master.quit()
        first1.quit()
        second1.quit()
        first1.destroy()

    if count1 == 2:
        selected_list = str(variable_second1.get())
        saveNewOrder(last_requested, selected_list)
        # Close tkinter dialogue box
        master.quit()
        first1.quit()
        second1.quit()
        second1.destroy()


# Function to prompt and list different food description for user to select from
def newOrderList(lists):
    global variable_first1
    global variable_second1
    global count1
    global first1
    global second1
    global dont_repeat1

    try:
        count1 = count1 + 1
        if count1 == 3:
            count1 = 1
        if count1 == 2:
            second1 = tkinter.Tk()
            # creating drop down
            variable_second1 = StringVar(second1)
            variable_second1.set("Please Click Here To Select")  # default value

            # Tkinter dialogue settings
            second1.title('Select Your Order Choice Below')
            second1.geometry("700x60")

            # Tkinter drop list settings
            w = tkinter.OptionMenu(second1, variable_second1, *lists)
            w.config(width=60)
            w.pack()

            # Tkinter button settings
            b = tkinter.Button(second1, text="Click To Proceed", width=20, command=proceedOrder)
            b.pack()

            print(str(food_descriptions).upper())
            SpeakText("ACCORDING TO OUR DATABASE, YOU HAVE NOT PURCHASED " + last_requested
                      + " BEFORE. PLEASE SELECT YOUR CHOICE USING THE DROP DOWN LIST TO PROCEED.")
            # Start the program loop
            second1.mainloop()

        if count1 == 1:
            if dont_repeat1 == 0:
                dont_repeat1 = 1
                master.quit()
                first1.quit()
                second1.quit()
                second1.destroy()
            first1 = tkinter.Tk()
            # creating drop down
            variable_first1 = StringVar(first1)
            variable_first1.set("Please Click Here To Select")  # default value
            # Tkinter dialogue settings
            first1.title('Select Your Order Choice Below')
            first1.geometry("700x60")

            # Tkinter drop list settings
            w = tkinter.OptionMenu(first1, variable_first1, *lists)
            w.config(width=60)
            w.pack()

            # Tkinter button settings
            b = tkinter.Button(first1, text="Click To Proceed", width=20, command=proceedOrder)
            b.pack()

            print(str(food_descriptions).upper())
            SpeakText("ACCORDING TO OUR DATABASE, YOU HAVE NOT PURCHASED " + last_requested
                      + " BEFORE. PLEASE SELECT YOUR CHOICE USING THE DROP DOWN LIST TO PROCEED.")
            # Start the program loop
            first1.mainloop()

    except Exception as e2:
        print("ERROR (NEW ORDER LIST): ", e2)
        SpeakText("PLEASE YOU NEED TO RESTART APPLICATION TO CHOOSE A NEW ORDER CHOICE")


# Function to get the most frequently bought food type based on the food to be bought
def getRegularOrders():
    global finalOrderCheck
    global order_confirmation
    global food_descriptions
    global Food_Category
    global Food_Description
    global newResponse
    global Order_Time
    global Order_Date
    global Order_Frequency
    global new_rows

    try:
        response = requests.get("https://akifagoelectronics.com/UoW_2065655_Project/5@VIoTphp2Files/checkOrders.php?NI_Number=" + ni_number +"&Food_Category=" + last_requested, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
        })
        # Check for http OK response code
        if response.status_code == 200:
            try:
                # Get JSON Objects form response text
                orders = json.loads(response.text)
                prev_frequency: int = 0
                # Loop through all received JSON Objects
                for order in orders:
                    # Get saved server questions and convert it to upper case
                    if int(order['Order_Frequency']) > prev_frequency:
                        Food_Category = str(order['Food_Category']).upper()
                        Food_Description = str(order['Food_Description']).upper()
                        Order_Time = str(order['Order_Time']).upper()
                        Order_Date = str(order['Order_Date']).upper()
                        Order_Frequency = str(order['Order_Frequency']).upper()

                        newResponse = "DO YOU WANT YOUR USUAL, WHICH IS " + Food_Description + "?." \
                                      + " ACCORDING TO OUR DATABASE, THE LAST TIME YOU ORDERED " \
                                      + Food_Category + " WAS ON " + Order_Date + " AND IT'S THE " \
                                      + str(Order_Frequency) \
                                      + " TIME YOU PLACED THE ORDER WHICH IS THE HIGHEST ORDER MADE FOR " \
                                      + Food_Category \
                                      + ". SHOULD I PROCEED WITH THE ORDER? PLEASE RESPOND WITH A YES OR A NO."

                        prev_frequency = int(order['Order_Frequency'])
                print("DO YOU WANT YOUR USUAL, WHICH IS " + Food_Description + "?."
                      + " ACCORDING TO OUR DATABASE, THE LAST TIME YOU ORDERED "
                      + Food_Category + " WAS ON " + Order_Date + " AND IT'S THE " + str(Order_Frequency)
                      + " TIME YOU PLACED THE ORDER WHICH IS THE HIGHEST ORDER MADE FOR " + Food_Category
                      + ". SHOULD I PROCEED WITH THE ORDER? PLEASE RESPOND WITH A YES OR A NO.")

                finalOrderCheck = 1
                order_confirmation = 0
                SpeakText(newResponse)
            except ValueError:
                try:
                    # Check for command keywords
                    if food_data['Category'].str.contains(last_requested).any():
                        new_rows = food_data.loc[(food_data['Category'] == last_requested), :]
                        if len(new_rows) > 0:
                            food_descriptions = new_rows['Description']

                            newOrderList(food_descriptions.tolist())
                        else:
                            print("SELECTED DATA ROW IS EMPTY FOR " + last_requested.upper())
                    else:
                        print("SORRY", last_requested, "FOOD ITEM NOT FOUND ON THE LIST")

                except ValueError as e2:
                    print("SELECTED DATA ROW IS EMPTY FOR " + last_requested.upper())
        else:
            print('Not Connected!')
    except requests.exceptions.RequestException as e1:
        print(e1)


# Function to save and confirm a new response given to the chatbot based on a question asked
def saveResponse(value):
    global recogSpeech
    try:
        response = requests.get("https://www.akifagoelectronics.com/robot@AE2527/newUpdate.php?question=" + recogSpeech + "&response=" + value, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
        })
        # Check for http OK response code
        if response.status_code == 200:
            # Get JSON Objects form response text
            SpeakText("I just learnt a new response today, thanks for teaching me.")

        else:
            print('Not Connected!')
    except requests.exceptions.RequestException as e:
        print(e)


# Function to get and say system time
def sayTime():
    today = datetime.today()
    hourValue = today.hour
    minuteValue = today.minute
    amPM: str = ""

    if hourValue >= 12:
        amPM = "PM"
    if 0 <= hourValue < 12:
        amPM = "AM"

    hourInt: int = int(hourValue)
    minuteInt: int = int(minuteValue)

    if hourInt > 12:
        hourInt = hourInt - 12
    if 0 < minuteInt < 10:
        minuteValue = "0" + str(minuteInt)
    if 0 < hourInt < 10:
        hourValue = "0" + str(hourInt)

    print(hourValue, ":", minuteValue, amPM)

    if minuteInt == 0:
        SpeakText("The time is " + str(hourValue) + " O clock." + amPM)
    if 0 < minuteInt < 30:
        SpeakText("The time is " + str(minuteValue) + "minutes pass " + str(hourValue) + amPM)
    if minuteInt == 30:
        SpeakText("The time is " + str(hourValue) + " thirty." + amPM)
    if minuteInt > 30:
        minuteInt = 60 - minuteInt
        hourInt = hourInt + 1
        SpeakText("The time is " + str(minuteInt) + " To. " + str(hourInt) + amPM)


# function to get user uploaded food image file type and perform food prediction
def uploadImage():
    global image_prediction
    global new_preRequested
    global suggested_food
    global scan_categories
    global new_value
    global read_labels
    global count

    # Get selected image path
    path = askopenfilename(filetypes=[("Image File", '.jpg')])
    print("SELECTED IMAGE PATH: ", path.upper())
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(path)

    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array
    # Predict meal routine
    if image_prediction == 1:
        # run the inference
        prediction = model_food.predict(data)
        print("FOOD MODEL PREDICTIONS: ", prediction)

        # to get the maximum predicted index position
        predicted_index = prediction.argmax()

        # read labels text file into list
        read_labels = open('labels.txt', 'r')
        labels_lists = read_labels.readlines()
        print("FOOD LABELS: ", ' '.join(labels_lists).replace("\n", ",").upper())

        # get the predicted class name
        predicted_class_name = labels_lists[predicted_index].replace("\n", "")

        # remove index value
        if " " in predicted_class_name:
            predicted_class_name = predicted_class_name.split(" ")[1]

        if "_" in predicted_class_name:
            predicted_class_name = predicted_class_name.replace("_", " ")

        print("FOOD PREDICTED CLASS NAME: ", predicted_class_name.upper())

        # Scan if predicted food item is in dataset
        new_preRequested = ""
        new_food_class = predicted_class_name.upper()

        scan_categories = food_data['Category'].tolist()
        for b in range(len(scan_categories)):
            new_value = scan_categories[b]
            if new_food_class == new_value:
                new_preRequested = new_value

        if len(new_preRequested) > 0:
            # Say the predicted food
            SpeakText("THE TRAINED KERAS MODEL WAS ABLE TO PREDICT " + new_preRequested)
            print("THE TRAINED KERAS MODEL WAS ABLE TO PREDICT: ", new_preRequested)

            # Food Suggestion Routine
            try:
                response = requests.get("https://akifagoelectronics.com/UoW_2065655_Project/5@VIoTphp2Files/checkCholesterol.php?NI_Number=" + ni_number, headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
                })
                # Check for http OK response code
                if response.status_code == 200:
                    # Get JSON Objects form response text
                    user = json.loads(response.text)
                    cholesterol_level: int = int(user[0]['Cholesterol'])
                    print("YOUR CHOLESTEROL LEVEL: ", cholesterol_level)
                    # Normal Cholesterol Level Is < 200
                    if cholesterol_level >= 200 or heart_condition == 1:
                        try:
                            # Check for command keywords
                            if food_data['Category'].str.contains(new_preRequested).any():
                                # get food description values using the requested name
                                description_rows = food_data.loc[(food_data['Category'] == new_preRequested),
                                                                 'Description']
                                # get food cholesterol values using the requested name
                                cholesterol_rows = food_data.loc[(food_data['Category'] == new_preRequested),
                                                                 'Data.Cholesterol']
                                # get the least cholesterol index from the selected dataframe
                                cholesterol_array = cholesterol_rows.to_numpy()
                                least_cholesterol_index = np.argmin(cholesterol_array)
                                # get the least cholesterol value from the selected dataframe
                                least_cholesterol = cholesterol_array[least_cholesterol_index]
                                # get the nth item from the dataframe
                                suggested_food = ' '.join(description_rows.take([least_cholesterol_index]))
                                if len(suggested_food) > 0:
                                    print("SUGGESTED FOOD: ", str(suggested_food).upper(),
                                          " || FOOD CHOLESTEROL CONTENT: ", least_cholesterol)
                                    # Convert text to speech
                                    SpeakText("MAY I SUGGEST THAT THIS TIME YOU EAT " + str(suggested_food)
                                              + " BECAUSE YOUR CHOLESTEROL LEVEL IS " + str(cholesterol_level)
                                              + " WHICH IS NOT GOOD.")
                                else:
                                    print("SELECTED DATA ROW IS EMPTY")
                            else:
                                print("SORRY", new_preRequested, "FOOD ITEM NOT FOUND ON THE LIST")

                        except ValueError as e2:
                            SpeakText("VALUE ERROR. UNABLE TO VERIFY FOOD CHOLESTEROL CONTENT NOW")
                    else:
                        SpeakText("YOUR CHOLESTEROL LEVEL IS GOOD. ENJOY YOUR FOOD.")
                else:
                    print('NOT CONNECTED!')
            except requests.exceptions.RequestException as e1:
                print(e1)
        else:
            print("SORRY PREDICTED FOOD ITEM NOT FOUND ON THE LIST")

    # Predict heart routine
    elif image_prediction == 2:
        # run the inference
        prediction = model_food.predict(data)
        print("HEART DISEASE MODEL PREDICTIONS: ", prediction)

        # to get the maximum predicted index position
        predicted_index = prediction.argmax()

        # read labels text file into list
        read_labels = open('labels.txt', 'r')
        labels_lists = read_labels.readlines()
        print("HEART DISEASE LABELS: ", ' '.join(labels_lists).replace("\n", ",").upper())

        # get the predicted class name
        predicted_class_name = labels_lists[predicted_index]
        if " " in predicted_class_name:
            predicted_class_name = predicted_class_name.split(" ")[1]
        print("HEART DISEASE PREDICTED CLASS NAME: ", predicted_class_name.upper())

    # Reset variable to 0
    image_prediction = 0


def endDialogue():
    global first
    global second
    if count == 1:
        master.quit()
        second.quit()
        first.quit()
        first.destroy()
    if count == 2:
        master.quit()
        second.quit()
        first.quit()
        second.destroy()
    uploadImage()


# function to prompt user to select and upload a food image file to predict
def predictMeal():
    global count
    global first
    global second
    global dont_repeat
    try:
        count = count + 1
        if count == 3:
            count = 1
        if count == 2:
            second = tkinter.Tk()

            # Tkinter button settings
            button = tkinter.Button(second, text="Click To Upload Image", width=20, command=endDialogue)
            button.pack()

            print("PLEASE CLICK TO UPLOAD A NEW IMAGE TO PREDICT")
            SpeakText("PLEASE CLICK TO UPLOAD A NEW IMAGE TO PREDICT")

            second.mainloop()
        if count == 1:
            if dont_repeat == 0:
                dont_repeat = 1
                second.destroy()
            first = tkinter.Tk()

            # Tkinter button settings
            button = tkinter.Button(first, text="Click To Upload Image", width=20, command=endDialogue)
            button.pack()

            print("PLEASE CLICK TO UPLOAD A NEW IMAGE TO PREDICT")
            SpeakText("PLEASE CLICK TO UPLOAD A NEW IMAGE TO PREDICT")

            first.mainloop()

    except Exception as e2:
        print("ERROR (PREDICT MEAL): ", e2)
        SpeakText("PLEASE YOU NEED TO RESTART APPLICATION TO PREDICT A NEW FOOD IMAGE")


# function to prompt user to select and upload a scanned heart image file to predict
def predictHeart():
    global count
    global first
    global second
    global dont_repeat
    try:
        count = count + 1
        if count == 3:
            count = 1
        if count == 2:
            second = tkinter.Tk()

            # Tkinter button settings
            button = tkinter.Button(second, text="Click To Upload Image", width=20, command=endDialogue)
            button.pack()

            print("PLEASE CLICK TO UPLOAD A NEW IMAGE TO PREDICT")
            SpeakText("PLEASE CLICK TO UPLOAD A NEW IMAGE TO PREDICT")

            second.mainloop()
        if count == 1:
            if dont_repeat == 0:
                dont_repeat = 1
                second.destroy()
            first = tkinter.Tk()

            # Tkinter button settings
            button = tkinter.Button(first, text="Click To Upload Image", width=20, command=endDialogue)
            button.pack()

            print("PLEASE CLICK TO UPLOAD A NEW IMAGE TO PREDICT")
            SpeakText("PLEASE CLICK TO UPLOAD A NEW IMAGE TO PREDICT")

            first.mainloop()

    except Exception as e2:
        print("ERROR (PREDICT HEART): ", e2)
        SpeakText("PLEASE YOU NEED TO RESTART APPLICATION TO PREDICT A NEW FOOD IMAGE")


# function to get system date
def sayDate():
    today = datetime.today()
    d2 = today.strftime("%B %d, %Y")
    print(d2)
    SpeakText(d2)


# function to say current weather condition from api call
def sayWeather(city):
    global process
    # base URL
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather?"
    # City Name CITY = "Hyderabad"
    # API key API_KEY = "Your API Key"
    # upadting the URL
    URL = BASE_URL + "q=" + city + "&units=metric" + "&appid=338a9e7e265bb9ed5f2a64599a020945"
    # HTTP request
    response = requests.get(URL)
    # checking the status code of the request
    if response.status_code == 200:
        # getting data in the json format
        data = response.json()
        # getting the main dict block
        main = data['main']
        # getting temperature
        temperature = main['temp']
        # getting the humidity
        humidity = main['humidity']
        # getting the pressure
        pressure = main['pressure']
        # weather report
        report = data['weather']
        print(f"{city:-^30}")
        print(f"Temperature: {temperature}")
        print(f"Humidity: {humidity}")
        print(f"Pressure: {pressure}")
        print(f"Weather Report: {report[0]['description']}")
        process = 0
        # Output the speech
        SpeakText("The weather report for " + city + " is as follows. The temperature is " + str(temperature)
                  + " degree celsius. The humidity is " + str(humidity) + "%. The pressure is " + str(pressure)
                  + " millibars. The weather report is " + report[0]['description'])
    else:
        # showing the error message
        print("Error in the HTTP request")

def exit_app():
    SpeakText("Thank you and have a nice day.")
    import main
    sys.exit()

# Loop infinitely for user to
# speak=
while (1):
    # Exception handling to handle
    # exceptions at the runtime
    try:
        # use the microphone as source for input.
        with sr.Microphone() as source2:
            # Check if national insurance number is saved
            if ni_number == "" or len(ni_number) == 0:
                my_file = Path('NI_Data.txt')
                if my_file.exists():
                    # read labels text file into list
                    read_labels = open('NI_Data.txt', 'r')
                    ni_number = read_labels.readline()
                else:
                    requestNI()

            if first_start == 0:
                first_start = 1
                SpeakText("THANKS FOR YOUR PATIENCE. "
                          + "ONE. TO PURCHASE FOOD: PLEASE SAY, I WANT TO BUY PIZZA, OR I WANT TO PURCHASE PIZZA. "
                          + "TWO. TO PREDICT FOOD: PLEASE SAY, I WANT TO PREDICT FOOD. "
                          + "THREE. TO CONFIRM HEART DISEASE: PLEASE SAY, PREDICT HEART. "
                          + "FOUR. TO CHECK THE WEATHER: PLEASE SAY, WHAT IS THE WEATHER?")

            # Monitor no response for 3 times and exit
            if exit_count == 3:
                exit_app()
                break

            # wait for a second to let the recognizer
            # adjust the energy threshold based on
            # the surrounding noise level
            r.adjust_for_ambient_noise(source2, duration=3)

            print("TALK TO ME, I'M LISTENING TO YOU.")

            # listens for the user's input
            audio2 = r.listen(source2)

            # Using google to recognize audio
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()

            print("RECOGNIZED SPEECH TEXT: ", str(MyText).upper())
            errorState = 0

            # Responses
            if noResponse == 1 and str(MyText).upper() != "DON'T BOTHER":
                noResponse = 0
                confirmAnswer = 1
                newResponse = MyText
                SpeakText("Did you say " + MyText)
                # Correct response
            elif confirmAnswer == 1 and noResponse == 0 and str(MyText).upper() == "YES" \
                    or confirmAnswer == 1 and noResponse == 0 and "YES" in str(MyText).upper() \
                    or confirmAnswer == 1 and noResponse == 0 and str(MyText).upper() == "YEAH" \
                    or confirmAnswer == 1 and noResponse == 0 and "YEAH" in str(MyText).upper():
                SpeakText("OK THEN")
                if newResponse != "":
                    saveResponse(str(newResponse).upper())
                # Repeats the new response saving
            elif confirmAnswer == 1 and noResponse == 0 and str(MyText).upper() == "NO" \
                    or confirmAnswer == 1 and noResponse == 0 and "NO" in str(MyText).upper():
                noResponse = 1
                confirmAnswer = 0
                newResponse = ""
                SpeakText("PLEASE REPEAT WHAT YOU SAID EARLIER.")
                # Reset to default
            elif noResponse == 1 and str(MyText).upper() == "DON'T BOTHER":
                noResponse = 0
                confirmAnswer = 0
                SpeakText("OK THANKS")
            # Time and Date Request Routine
            elif str(MyText).upper() == "WHAT IS THE TIME" or str(MyText).upper() == "WHAT IS THE TIME NOW" \
                    or "THE TIME NOW" in str(MyText).upper() or "TIME NOW" in str(MyText).upper():
                order_confirmation = 0
                confirm_suggestion = 0
                finalOrderCheck = 0
                sayTime()
            elif str(MyText).upper() == "WHAT IS THE DATE" or str(MyText).upper() == "WHAT IS THE DATE NOW" \
                    or "THE DATE NOW" in str(MyText).upper() or "DATE NOW" in str(MyText).upper():
                order_confirmation = 0
                confirm_suggestion = 0
                finalOrderCheck = 0
                sayDate()
            # Image Prediction
            elif str(MyText).upper() == "PREDICT FOOD" or str(MyText).upper() == "PREDICT FOOD" \
                    or "FOOD" in str(MyText).upper() or "FOOD" in str(MyText).upper():
                image_prediction = 1
                predictMeal()
            # Using DL
            elif str(MyText).upper() == "PREDICT HEART IMAGE" or str(MyText).upper() == "PREDICT HEART IMAGE" \
                    or "HEART IMAGE" in str(MyText).upper() or "HEART IMAGE" in str(MyText).upper():
                image_prediction = 2
                predictHeart()
            # Using KNN
            elif str(MyText).upper() == "PREDICT HEART" or str(MyText).upper() == "PREDICT HEART" \
                    or "HEART" in str(MyText).upper() or "HEART" in str(MyText).upper():
                predictHeartDisease()
            # Weather Request Routine
            elif str(MyText).upper() == "WHAT IS THE WEATHER" or str(MyText).upper() == "WHAT IS THE WEATHER NOW" \
                    or "THE WEATHER" in str(MyText).upper():
                order_confirmation = 0
                confirm_suggestion = 0
                finalOrderCheck = 0
                noResponse = 0
                confirmAnswer = 0
                process = 1
                SpeakText("Please which city are you currently")
            # Food Order Routine
            elif order_confirmation == 1 and str(MyText).upper() == "YES" \
                    or finalOrderCheck == 1 and str(MyText).upper() == "YES" \
                    or confirm_suggestion == 1 and str(MyText).upper() == "YES" \
                    or order_confirmation == 1 and "YES" in str(MyText).upper()\
                    or finalOrderCheck == 1 and "YES" in str(MyText).upper()\
                    or confirm_suggestion == 1 and "YES" in str(MyText).upper() \
                    or order_confirmation == 1 and str(MyText).upper() == "YEAH" \
                    or finalOrderCheck == 1 and str(MyText).upper() == "YEAH" \
                    or confirm_suggestion == 1 and str(MyText).upper() == "YEAH" \
                    or order_confirmation == 1 and "YEAH" in str(MyText).upper() \
                    or finalOrderCheck == 1 and "YEAH" in str(MyText).upper() \
                    or confirm_suggestion == 1 and "YEAH" in str(MyText).upper():
                SpeakText("OK THEN")
                if last_requested != "" and len(ni_number) > 0:
                    print("CONFIRMING YOUR ORDER...")
                    if order_confirmation == 1:
                        getRegularOrders()
                    elif finalOrderCheck == 1:
                        getCholesterolLevel()
                    elif confirm_suggestion == 1:
                        # Uploads Suggested Food Item To Server
                        saveNewOrder(last_requested, suggested_food)
            elif order_confirmation == 1 and str(MyText).upper() == "NO" \
                    or finalOrderCheck == 1 and str(MyText).upper() == "NO" \
                    or confirm_suggestion == 1 and str(MyText).upper() == "NO" \
                    or order_confirmation == 1 and "NO" in str(MyText).upper()\
                    or finalOrderCheck == 1 and "NO" in str(MyText).upper()\
                    or confirm_suggestion == 1 and "NO" in str(MyText).upper():
                if finalOrderCheck == 1:
                    try:
                        # Check for command keywords
                        if food_data['Category'].str.contains(last_requested).any():
                            new_rows = food_data.loc[(food_data['Category'] == last_requested), :]
                            if len(new_rows) > 0:
                                food_descriptions = new_rows['Description']

                                newOrderList(food_descriptions.tolist())
                            else:
                                print("SELECTED DATA ROW IS EMPTY FOR " + last_requested.upper())
                        else:
                            print("SORRY", last_requested, "FOOD ITEM NOT FOUND ON THE LIST")

                    except ValueError as e:
                        print("SELECTED DATA ROW IS EMPTY FOR " + last_requested.upper())
                    finalOrderCheck = 0
                elif order_confirmation == 1:
                    order_confirmation = 0
                    print("SORRY ABOUT THAT. PLEASE SAY, FOR EXAMPLE, I WANT TO BUY PIZZA, OR I WANT TO PURCHASE PIZZA")
                    SpeakText(
                        "SORRY ABOUT THAT. PLEASE SAY, FOR EXAMPLE, I WANT TO BUY PIZZA, OR I WANT TO PURCHASE PIZZA")
                else:
                    confirm_suggestion = 0
                    print("SORRY WE CANNOT PROCEED WITH THE ORDER NOW. PLEASE VISIT THE DOCTOR FOR FURTHER ADVICE.")
                    SpeakText("SORRY WE CANNOT PROCEED WITH THE ORDER NOW. PLEASE VISIT THE DOCTOR FOR FURTHER ADVICE.")
            elif "BUY" in str(MyText).upper() or "PURCHASE" in str(MyText).upper()  \
                    or "I NEED" in str(MyText).upper():
                last_requested = ""
                order_confirmation = 1
                split_speech = str(MyText).upper().split(" ")
                scan_categories = food_data['Category'].tolist()
                for i in range(len(scan_categories)):
                    new_value = scan_categories[i]
                    for new_data in split_speech:
                        if new_value == new_data:
                            last_requested = new_data
                if len(last_requested) > 0:
                    print("NEW CUSTOMER REQUEST: ", last_requested)
                    prev_command = str(MyText).upper()
                    SpeakText("DID YOU SAY: " + str(MyText).upper())
            # Get Weather Routine
            elif process == 1 and str(MyText).upper() != "DON'T BOTHER":
                process = 2
                cityResponse = MyText
                SpeakText("Did you say " + MyText)
            elif process == 1 and str(MyText).upper() == "DON'T BOTHER":
                process = 0
                cityResponse = ""
                SpeakText("OK THANKS")
            elif process == 2 and str(MyText).upper() == "YES" \
                    or process == 2 and "YES" in str(MyText).upper() \
                    or process == 2 and str(MyText).upper() == "YEAH" \
                    or process == 2 and "YEAH" in str(MyText).upper():
                SpeakText("OK THEN")
                if cityResponse != "":
                    sayWeather(str(cityResponse).upper())
            elif process == 2 and str(MyText).upper() == "NO" \
                    or process == 2 and "NO" in str(MyText).upper():
                process = 1
                cityResponse = ""
                SpeakText("Please which city are you currently")
            # Checks Database For Response to a Question Asked
            elif "WHICH" in str(MyText).upper() or "HOW" in str(MyText).upper()\
                    or "WHY" in str(MyText).upper() or "WHERE" in str(MyText).upper() \
                    or "WHEN" in str(MyText).upper() or "DEFINE" in str(MyText).upper() \
                    or "WHAT" in str(MyText).upper() or "ARE YOU" in str(MyText).upper() \
                    or "DO YOU" in str(MyText).upper() or "WHO" in str(MyText).upper():
                order_confirmation = 0
                confirm_suggestion = 0
                finalOrderCheck = 0
                # Parse converted text to the declared function
                getResponse(MyText)

    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    except sr.UnknownValueError:
        errorState = 1
        if process == 1:
            SpeakText("PLEASE REPEAT THE SAME ANSWER.")
        else:
            SpeakText("TALK TO ME, I'M LISTENING TO YOU.")
            exit_count = exit_count + 1

