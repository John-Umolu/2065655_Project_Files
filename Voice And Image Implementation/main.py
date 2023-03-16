import sys
import requests
import tkinter
from tkinter import *
import pyttsx3

# Create a new Tkinter Form
# Tkinter dialogue settings
master1 = tkinter.Tk()
master1.resizable(False, False)
master1.geometry("450x80")
master1.title("Voice Based Diet Recommendation By 2065655")
master1.configure(bg='brown')

# assign the first label
l1 = Label(text="Please Enter Your Hospital/NI Number", fg="white", bg="grey", font=('Times', 14))
# create the user text input box
inputtxt = Entry(master1, width=40, bg="white", justify=CENTER, font=('Times', 12))

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
def saveNI(id):
    # Gets the new value in the tkinter text box
    with open("NI_Data.txt", "w") as text_file:
        text_file.write(id)

    master1.quit()
    master1.destroy()
    import virtualHealthBot
    sys.exit()

# Function to check question and retrieve a response saved on the database
def checkUser(value):

    try:
        response = requests.get("https://akifagoelectronics.com/UoW_2065655_Project/5@VIoTphp2Files/verifyNI.php?NI_Number=" + value, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
        })
        # Check for http OK response code
        if response.status_code == 200:
            # Get JSON Objects form response text
            json_response = str(response.text).upper()
            if "VALID" == json_response:
                print("VALID USER")
                SpeakText("CONGRATS! YOU ARE REGISTERED.")
                saveNI(value)
            else:
                print("NOT VALID USER")
                SpeakText("SORRY! YOU ARE NOT REGISTERED FOR THIS SERVICE. PLEASE VISIT THE NEAREST MEDICAL CENTER.")
        else:
            print('Not Connected!')
    except requests.exceptions.RequestException as e1:
        print(e1)


def proceed():
    entered_ni = str(inputtxt.get())
    if len(entered_ni) > 0:
        checkUser(entered_ni)

# Tkinter textbox settings
# Create a text box to enter NI number
l1.pack()
inputtxt.pack()
inputtxt.focus_set()

# Tkinter button settings
b = Button(master1, height=1, width=15, text="Click To Proceed",
           command=lambda: proceed(), fg="white", bg="black", font=('Times', 14))
b.pack()

print("WELCOME TO STAY HEALTHY VIRTUAL ASSISTANCE. PLEASE ENTER YOUR HOSPITAL NUMBER")
SpeakText("WELCOME TO STAY HEALTHY VIRTUAL ASSISTANCE. PLEASE ENTER YOUR HOSPITAL NUMBER")

# Bind the Enter Key to the window
master1.bind('<Return>', lambda event=None: b.invoke())

# Start the program loop
master1.mainloop()

