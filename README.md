# 7CS042/UM2: MSc Project Artificial Intelligence By Umolu John Chukwuemeka (2065655) <br>
The project developed a web-based diet recommendation application that suggests food items for heart disease patients based on the predicted outcome of the web-based heart disease prediction app. <br>
# Heart Disease Prediction App <br>
The web-based heart disease prediction application is accessed using the below link. <br>
URL link: https://www.akifagoelectronics.com/2065655_Hospital/ <br>
The first page will request the user to enter the patient’s information for the heart disease prediction, as shown below. <br>
![h1](https://user-images.githubusercontent.com/106328663/235750177-6c4dca73-e1da-4661-80f2-9af3734c5762.png) <br>
The patient’s information was entered into the webpage form, and the Submit button was clicked for the heart disease prediction process, as shown in the image above. The trained machine learning model was used to perform the prediction, and the predicted outcome was then uploaded and stored on the database table, as shown below. <br>
![h2](https://user-images.githubusercontent.com/106328663/235750263-74c0ab59-7fe1-4b23-912d-a796604c2818.png) <br>
The six machine learning models were trained on the web server using the Train Models buttons, as shown in the image above. <br>
The below result page is displayed when the Train Models (Without Using Bagging and AdaBoost) button is clicked on the above webpage. <br>
![heart disease prediction 2](https://user-images.githubusercontent.com/106328663/225674803-56958c9e-9a72-4f16-942f-6d07715bb11d.png) <br>

# The Diet Recommendation App <br>
The web-based diet recommendation application is accessed using the below link. <br>
URL link: https://www.akifagoelectronics.com/2065655_Home/ <br>
The first page asked the user to enter their Hospital Number, as shown below. <br>
![d1](https://user-images.githubusercontent.com/106328663/235749551-e40881ef-e8b6-4627-974b-ce49ea7e912d.png) <br>
Please Note: The hospital number required in the above image is the same as the number used while filling out the patient’s form on the heart disease prediction web application. <br>
If the user is registered to use the web application, a webpage will display, requesting the user to enter the food item they want to buy, as shown in the image below. <br>
![d2](https://user-images.githubusercontent.com/106328663/235749637-7df4e8a5-e701-45a4-92ce-85b12f5b8067.png) <br>
After entering the name of the food item, the Proceed to Buy button is then clicked, and the GET request is sent to the webserver to check the user’s saved medical record on the database to verify the last saved heart disease predicted outcome on the database using the user’s hospital number initially entered on the start page. A response is received, and if the data suggests that the user is prone to heart disease or has very high cholesterol in the body, the below webpage will be displayed, showing details received from the web server, as shown below. <br>
![d3](https://user-images.githubusercontent.com/106328663/235749776-e78367fe-1c4f-44d7-a116-7140e5be820a.png) <br>
The user also has the option to view available stores on a Google map interface, as shown in the image below. <br>
![d4](https://user-images.githubusercontent.com/106328663/235749838-c5fa214b-a4f3-4a5f-91f8-b742737cd901.png) <br>


