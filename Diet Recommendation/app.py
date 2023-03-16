from flask import Flask, redirect, session, url_for, request, render_template
import requests
import json
import simplejson
import pandas as pd
import numpy as np
from flask_googlemaps import GoogleMaps
from flask_googlemaps import Map
import geocoder

app = Flask(__name__)

# Details on the Secret Key: https://flask.palletsprojects.com/en/1.1.x/config/#SECRET_KEY
# NOTE: The secret key is used to cryptographically-sign the cookies used for storing
#       the session data.
app.secret_key = 'Enter App Secret Key Here'  # Note the key used was removed to prevent public use

# you can set key as config
api_key = 'Enter Google API Key Here'  # Note the key used was removed to prevent public use

# set api_key
# get api key from Google API Console (https://console.cloud.google.com/apis/)
GoogleMaps(app, key=api_key)

ni_number = ""
drop_list = []
selected_food = ""
food = ""
all_suggestions = []
Food_Description = ""

# Read Food Dataset From Webserver
food_data = pd.read_csv('food.csv')

# drop rows with null values
food_data.dropna()

# Change food category names to upper case
food_data['Category'] = food_data['Category'].str.upper()

@app.route('/')
def home():
	return render_template('main.html')
	
@app.route('/success/<name>')
def success(name):
    global ni_number
    try:
        response = requests.get("https://akifagoelectronics.com/UoW_2065655_Project/5@VIoTphp2Files/verifyNI.php?NI_Number=" + name, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
        })
        # Check for http OK response code
        if response.status_code == 200:
            # Get JSON Objects form response text
            json_response = str(response.text).upper()
            if "VALID" == json_response:
                ni_number = str(name)
                session['userid'] = ni_number
                return render_template('order.html')
            else:
                return 'ATTENTION PLEASE!!! NO MEDICAL RECORD FOUND FOR : %s' % str(name).upper()
        else:
            return 'Not Connected!'
    except requests.exceptions.RequestException as e1:
        return 'Error:' + e1

   

@app.route('/main',methods = ['POST', 'GET'])
def main():
    if request.method == 'POST':
        user = request.form['nm']
        if len(user) > 0:
            return redirect(url_for('success',name = user))
        else:
            return "NO USER ENTRY FOUND, PLEASE ENTER A VALID HOSPITAL NUMBER."
    else:
        user = request.args.get('nm')
        if len(user) > 0:
            return redirect(url_for('success',name = user))
        else:
            return "NO USER ENTRY FOUND, PLEASE ENTER A VALID HOSPITAL NUMBER."
      
@app.route('/new_view/<item>')
def new_view(item):
    food = item
    all_suggestions = []
    
    # Check for food command keywords
    if food_data['Category'].str.contains(food.upper()).any():
            
        # get all the rows with the food title
        nfood_data = food_data.loc[(food_data['Category'].str.contains(food.upper())), :]
            
        # remove duplicated rows
        nfood_data = nfood_data.drop_duplicates(subset=['Data.Cholesterol'], keep='first')
                            
        #all_suggestions = nfood_data.drop(columns=['Category', 'Data.Alpha Carotene', 'Data.Beta Carotene', 'Data.Beta Cryptoxanthin', 'Data.Lutein and Zeaxanthin'])
        # rename column names
        nfood_data.columns = nfood_data.columns.str.replace('Description', 'Food Description')
        nfood_data.columns = nfood_data.columns.str.replace('Data.Cholesterol', 'Cholesterol')
        nfood_data.columns = nfood_data.columns.str.replace('Data.Sugar Total', 'Total Sugar')
        nfood_data.columns = nfood_data.columns.str.replace('Data.Fiber', 'Fiber')
        nfood_data.columns = nfood_data.columns.str.replace('Data.Protein', 'Protein')
        nfood_data.columns = nfood_data.columns.str.replace('Data.Fat.Saturated Fat', 'Saturated Fat')
        nfood_data.columns = nfood_data.columns.str.replace('Data.Vitamins.Vitamin C', 'Vitamin C')
        
        # select columns to display
        all_suggestions = nfood_data.loc[:,nfood_data.columns.isin(['Food Description','Cholesterol', 'Total Sugar', 'Fiber', 'Protein', 'Saturated Fat', 'Vitamin C'])]
        
        # convert list to dataframe                    
        df = pd.DataFrame(all_suggestions)
        
        # return dataframe table
        return render_template('food_discription.html',tables= [df.to_html(classes='data')],titles=['na','Food Composition Table For ' + str(food).capitalize()])
        
        #return df.to_html(header="true", table_id="table", index=False, justify = 'center')
        
    else:
        return str(food).upper() + " NOT FOUND ON THE FOOD LIST. PLEASE CHECK BACK LATER."

@app.route('/allfood',methods = ['POST', 'GET'])
def allfood():
    food = str(session['food'])
    if len(food) > 0:
        return redirect(url_for('new_view',item = food))
    else:
        return "OPPS! REQUEST NOT SUCCESSFUL AT THE MOMENT, PLEASE TRY AGAIN LATER."
            
@app.route('/new_order/<item>')
def new_order(item):
    global ni_number
    global food
    global Food_Description
    #global all_suggestions    
    ni_number = str(session['userid'])
    food = str(session['food'])
    if len(ni_number) <= 0:
        return "YOUR USER NUMBER WAS INVALID OR NOT REGISTERED FOR THIS SERVICE. PLEASE VISIT A MEDICAL CENTER OR TRY AGAIN."
    if len(item) > 0:
        food = item
        if len(food) <= 0:
            return "EMPTY OR NO VALID USER ENTRY FOUND. PLEASE ENTER A VALID FOOD ITEM NAME TO PROCEED. FOR EXAMPLE, TO PURCHASE PIZZA, ENTER PIZZA."
        try:
            response = requests.get("https://akifagoelectronics.com/UoW_2065655_Project/5@VIoTphp2Files/checkCholesterol.php?NI_Number=" + ni_number, headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"})
            # Check for http OK response code
            if response.status_code == 200:
                # Get JSON Objects form response text
                #user = json.loads(str(response.text))
                user = simplejson.loads(str(response.text))
                cholesterol_level = 0
                food_allergies = ""
                feedback = ""
                # Loop through all received JSON Objects
                for obj in user:
                    # Get cholesterol level
                    cholesterol_level = int(obj['Cholesterol'])
                    # get food allergies
                    food_allergies = str(obj['Food_Allergies'])
                    # get heart disease status
                    heart_status = int(obj['HeartDisease'])
                    
                # clear list
                drop_list = []
                if heart_status == 0:
                    drop_list.append("CONGRATULATION! YOU DONT HAVE HEART DISEASE. STAY HEALTHY.")
                if heart_status == 1:
                    drop_list.append("OOPS! YOU HAVE CHANCES OF HEART DISEASE. PLEASE CONTACT THE NEAREST HOSPITAL.")
                if cholesterol_level < 200:
                    drop_list.append("CHOLESTEROL LEVEL IN YOUR BODY: NORMAL.")
                if cholesterol_level >= 200:
                    drop_list.append("CHOLESTEROL LEVEL IN YOUR BODY: ABNORMAL.")
                
                drop_list.append("CHOLESTEROL VALUE: " + str(cholesterol_level) + ".") 
                drop_list.append("FOOD ALLERGIES: " + str(food_allergies.upper()) + ".")
                drop_list.append("I SUGGESTED SOME FOOD ITEMS WITH THE LEAST CHOLESTEROL VALUE BASED ON YOUR ORDER.")
                
                finalOrderCheck = 0
                # Normal Cholesterol Level Is < 200
                if cholesterol_level >= 200 or heart_status == 1:
                    # get the requested food item
                    try:
                        # Check for food command keywords
                        if food_data['Category'].str.contains(food.upper()).any():
            
                            # get all the rows with the food title
                            nfood_data = food_data.loc[(food_data['Category'].str.contains(food.upper())), :]
            
                            # remove duplicated rows
                            nfood_data = nfood_data.drop_duplicates(subset=['Data.Cholesterol'], keep='first')
                            
                            #all_suggestions = nfood_data.drop(columns=['Category'])
            
                            # get food description values using the requested name
                            description_rows = nfood_data['Description']
            
                            # get food cholesterol values using the requested name
                            cholesterol_array = nfood_data['Data.Cholesterol'].to_numpy()
            
                            # declare the variable
                            suggested_food = ""
                            chol_values = ""
            
                            if len(cholesterol_array) > 0 and len(cholesterol_array) <= 5:
                                # get the index position for the least cholesterol value
                                least_cholesterol_index = np.argmin(cholesterol_array)
            
                                # get the least cholesterol value from the selected dataframe
                                least_cholesterol = cholesterol_array[least_cholesterol_index]
                                
                                chol_values = str(least_cholesterol) + " milligrams (mg)."
                                drop_list.append("MAXIMUM CHOLESTEROL VALUE (SUGGESTED FOOD): " + str(chol_values).upper())
                                drop_list.append("PLEASE SELECT AN OPTION TO PLACE ORDER ONLINE OR CLICK TO VIEW AVAILABLE STORE AND RESTAURANT LOCATION.")
                
                                # get the nth item from the dataframe
                                drop_list.append(' '.join(description_rows.take([least_cholesterol_index])))
                
                            else:
                                if len(cholesterol_array) > 5:
                                    # get the least cholesterol value from the selected dataframe
                                    A, B, C, D, E = np.partition(cholesterol_array, 4)[0:5]
                                    
                                    chol_values = str(E) + " milligrams (mg)."
                                    drop_list.append("MAXIMUM CHOLESTEROL VALUE (SUGGESTED FOOD): " + str(chol_values).upper())
                                    drop_list.append("PLEASE SELECT AN OPTION TO PLACE ORDER ONLINE OR CLICK TO VIEW AVAILABLE STORE AND RESTAURANT LOCATION.")
                                    
                                    # find the index position for the least value
                                    # get the nth item from the dataframe
                                    drop_list.append(' '.join(description_rows.take([cholesterol_array.tolist().index(A)])))
                                    drop_list.append(' '.join(description_rows.take([cholesterol_array.tolist().index(B)])))
                                    drop_list.append(' '.join(description_rows.take([cholesterol_array.tolist().index(C)])))
                                    drop_list.append(' '.join(description_rows.take([cholesterol_array.tolist().index(D)])))
                                    drop_list.append(' '.join(description_rows.take([cholesterol_array.tolist().index(E)])))
                            
                            return render_template('final.html', foods=drop_list)
            
            
                        else:
                            return "SORRY " + food.upper() + " FOOD ITEM NOT FOUND ON THE LIST"

                    except ValueError as e:
                        return "SELECTED DATA ROW IS EMPTY FOR " + food.upper() + ". ERROR: " + str(e).upper()
                        
                else:
                    try:
                        response = requests.get("https://akifagoelectronics.com/UoW_2065655_Project/5@VIoTphp2Files/checkOrders.php?NI_Number=" + ni_number + "&Food_Category=" + food, headers={
                            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"})
                        # Check for http OK response code
                        if response.status_code == 200:
                            # Get JSON Objects form response text
                            orders = json.loads(response.text)
                            
                            prev_frequency = 0
                            # Loop through all received JSON Objects
                            for order in orders:
                                # Get saved server questions and convert it to upper case
                                if int(order['Order_Frequency']) > prev_frequency:
                                    Food_Category = str(order['Food_Category']).upper()
                                    Food_Description = str(order['Food_Description'])
                                    Order_Time = str(order['Order_Time']).upper()
                                    Order_Date = str(order['Order_Date']).upper()
                                    Order_Frequency = str(order['Order_Frequency']).upper()
                                    
                                    # save user frequently bought food record on session
                                    session['frequent_food'] = Food_Description
                                    
                                    newResponse = "DO YOU WANT YOUR USUAL, WHICH IS " + Food_Description.upper() + "?." \
                                        + " ACCORDING TO OUR DATABASE, THE LAST TIME YOU ORDERED " \
                                        + Food_Category + " WAS ON " + Order_Date + " AND IT'S THE " + str(Order_Frequency) \
                                        + " TIME YOU PLACED THE ORDER WHICH IS THE HIGHEST ORDER MADE FOR " + Food_Category \
                                        + ". DO YOU WANT TO PROCEED WITH THE ORDER?"

                                    prev_frequency = int(order['Order_Frequency'])
                        
                            feedback = []
                            feedback.append(str(drop_list[0]))
                            feedback.append(str(drop_list[1]))
                            feedback.append(str(drop_list[2]))
                            feedback.append(str(drop_list[3]))
                            #feedback.append("For more information and guidance about eating a healthy, balanced diet. Visit: https://www.nhs.uk/live-well/eat-well/")
                            feedback.append(str(newResponse))
                            
                            return render_template('final2.html', feedbacks=feedback)
                    
                        else:
                            return 'Not Connected!'
                                
                    except Exception as e1:
                        if 'JSON OBJECT' in str(e1).upper() and len(str(drop_list[0])) > 0:
                            # Check for food command keywords
                            if food_data['Category'].str.contains(food.upper()).any():
            
                                # get all the rows with the food title
                                nfood_data = food_data.loc[(food_data['Category'].str.contains(food.upper())), :]
            
                                # remove duplicated rows
                                nfood_data = nfood_data.drop_duplicates(subset=['Data.Cholesterol'], keep='first')
            
                                # get food description values using the requested name
                                description_rows = nfood_data['Description']
        
                                return render_template('final3.html', foods=description_rows)
        
                            else:
                                return str(food).upper() + " NOT FOUND ON THE FOOD LIST. PLEASE CHECK BACK LATER."
                        else:
                            return 'Error:' + str(e1).upper()
                
            else:
                return 'Not Connected!'
                
        except Exception as e1:
            return 'CONNECTION FAILED. PLEASE TRY AGAIN LATER. ERROR FOUND: ' + str(e1).upper()
        
@app.route('/order',methods = ['POST', 'GET'])
def order():
    if request.method == 'POST':
        food = request.form['nm']
        if len(food) > 0:
            session['food'] = food
            return redirect(url_for('new_order',item = food))
        else:
            return "EMPTY OR NO VALID USER ENTRY FOUND. PLEASE ENTER A VALID FOOD ITEM NAME TO PROCEED. FOR EXAMPLE, TO PURCHASE PIZZA, ENTER PIZZA."
    else:
        food = request.args.get('nm')
        if len(food) > 0:
            return redirect(url_for('new_order',item = food))
        else:
            return "EMPTY OR NO VALID USER ENTRY FOUND. PLEASE ENTER A VALID FOOD ITEM NAME TO PROCEED. FOR EXAMPLE, TO PURCHASE PIZZA, ENTER PIZZA."

# save new order to database table
@app.route('/saveNewOrder/<food_description>')
def saveNewOrder(food_description):
    ni_number = str(session['userid'])
    food = str(session['food'])
    try:
        response = requests.get("https://akifagoelectronics.com/UoW_2065655_Project/saveData.php?Food_NI_Number=" + str(ni_number)
                                + "&Food_Category=" + str(food) + "&Food_Description=" + str(food_description), headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
        })
        
        # Check for http OK response code
        if response.status_code == 200:
            # Get JSON Objects form response text
            return str("THANK YOU. YOUR ONLINE PURCHASE FOR " + str(food_description.replace('%20', ' ').replace('%2C', ',')).upper() + " WAS SUCCESSFUL.")
            
        else:
            return "ORDER MESSAGE: " + str('Not Connected!').upper()
            
    except requests.exceptions.RequestException as e:
        return "ORDER ERROR: " + str(e).upper()

@app.route('/onlinepurchase',methods = ['POST', 'GET'])
def onlinepurchase():
    select = request.form.get('comp_select')
    ni_number = str(session['userid'])
    food = str(session['food'])
    if len(select) > 0 and len(food) > 0 and len(ni_number) > 0:
        session['food_description'] = select
        return redirect(url_for('saveNewOrder',food_description = select))
    else:
        return "CANNOT PLACE NEW ORDER NOW, SOME PARAMETERS ARE NOT FOUND. PLEASE REPEAT PROCESS AGAIN."

@app.route('/yesreply',methods = ['POST', 'GET'])
def yesreply():
    ni_number = str(session['userid'])
    food = str(session['food'])
    Food_Description = str(session['frequent_food'])
    if len(Food_Description) > 0 and len(food) > 0 and len(ni_number) > 0:
        return redirect(url_for('saveNewOrder',food_description = Food_Description))
    else:
        return "CANNOT PLACE NEW ORDER NOW, SOME PARAMETERS ARE NOT FOUND. PLEASE REPEAT PROCESS AGAIN."

# load new food list
@app.route('/new_list/<food>')
def new_list(food):
   # Check for food command keywords
    if food_data['Category'].str.contains(food.upper()).any():
            
        # get all the rows with the food title
        nfood_data = food_data.loc[(food_data['Category'].str.contains(food.upper())), :]
            
        # remove duplicated rows
        nfood_data = nfood_data.drop_duplicates(subset=['Data.Cholesterol'], keep='first')
            
        # get food description values using the requested name
        description_rows = nfood_data['Description']
        
        return render_template('final3.html', foods=description_rows)
        
    else:
        return str(food).upper() + " NOT FOUND ON THE FOOD LIST. PLEASE CHECK BACK LATER." 
    
@app.route('/noreply',methods = ['POST', 'GET'])
def noreply():
    food = str(session['food'])
    if len(food) > 0:
        return redirect(url_for('new_list',food = food))
    else:
        return "OPPS! REQUEST NOT SUCCESSFUL AT THE MOMENT, PLEASE TRY AGAIN LATER."

# load new food list
@app.route('/store_location/<food>')
def store_location(food):
    # Read Store Dataset 
    store_data = pd.read_csv('stores.csv')

    # drop rows with null values
    store_data.dropna()

    # Change food category names to upper case
    store_data['category'] = store_data['category'].str.upper()
    
    # Check for food command keywords
    if not store_data['category'].str.contains(food.upper()).any():
        return "NO STORE LOCATION AVAILABLE FOR " + str(food.upper()) + ". PLEASE TRY AGAIN LATER."
                            
    # get all the rows with the food title
    nstore_data = store_data.loc[(store_data['category'].str.contains(food.upper())), :]
    
    # remove duplicated rows
    nstore_data = nstore_data.drop_duplicates(subset=['location'], keep='first')

    """Create map with all markers within bounds."""
    # long list of coordinates
    locations = []
    
    default_lat = float(51.509865)
    default_lng = float(-0.118092)
    
    names = nstore_data['name'].tolist()
    alllocations = nstore_data['location'].tolist()
    
    for i in range(len(names)):
        #mapinfo = ("<b>" + str(names[i]) + "</b>")
        mapinfo = str(names[i]).replace("'", "").strip()
        lat_lng = str(alllocations[i]).replace(' ', '').strip().split(',')
        new_lat = float(lat_lng[0])
        new_lng = float(lat_lng[1])
        locations.append({'lat': new_lat,'lng': new_lng,'infobox': mapinfo})
    
    map = Map(
        identifier="map",
        lat=default_lat,
        lng=default_lng,
        infobox = [(loc['infobox']) for loc in locations],
        markers={'http://maps.google.com/mapfiles/ms/icons/green-dot.png':[(loc['lat'],loc['lng'],str(loc['infobox'][0:6]) + "..") for loc in locations]},
        zoom = 18,
        fit_markers_to_bounds = True
    )
    return render_template('map.html', map=map)
    
@app.route("/locate" , methods=['GET', 'POST'])
def locate():
    global food
    food = str(session['food'])
    if len(food) > 0:
        return redirect(url_for('store_location',food = food))
    else:
        return "OPPS! REQUEST NOT SUCCESSFUL AT THE MOMENT, PLEASE TRY AGAIN LATER."

if __name__ == '__main__':
   app.run(debug = True)