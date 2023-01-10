Member 1: Manimaran Hendran 00811949

Member 2: Bachhav Yash 22109773

Title: Surviving the Unsinkable

Project Description:

    Undoubtedly one of the most well-known shipwrecks in history is the sinking of the Titanic.

    The RMS Titanic, which was widely believed to be "unsinkable," perished on April 15, 1912, after striking an iceberg while on her first voyage. Unfortunately, there were not enough lifeboats to accommodate everyone, and 1502 out of 2224 passengers and staff perished.


    Even while survival required a certain amount of luck, it appears that some groups of people had a higher chance of living than others.

    We will create a prediction model in this work to address the query, "What kinds of persons were more likely to survive?" using traveler data (ie name, age, gender, socio-economic class, etc).

    This widget will take the inputs from the user and will predict if they would have survived the Titanic disaster (based on input features) if they were onboard. 


Required Installations on the PC:

    1. PyCharm IDE

    2. Libraries to be installed in PyCharm:
        1. Matplotlib
        2. Pandas
        3. Scikit-Learn
        4. PyQt5
        

    3. Titanic - Machine Learning from Disaster Training & Testing Dataset (Source: Kaggle)
        Link to download the data: https://www.kaggle.com/c/titanic/data

    4. Download the repository from mygit and run it on the local computer to start the widget.

Basic Usage:

	1). Run the file 'widgets.py' on PyCharm (any IDE will work).

	2). A Window will appear asking the user to enter their Gender, Age, Fare and Passenger Class.

		i). You can choose Fare between 7 and 21 British Pounds.

		ii). There are 3 Passenger Classes: 1st Class, 2nd Class, 3rd Class. So enter 1, 2 or 3 accordingly.

	3). After entering all the details press the "Predict" button and you will know if you would have survived the Titanic Incident

	4). Please make sure, you only press predict button after entering all the details. Else the widget will crash.

	5). The Widget is designed in a way that it will only accept the numbers as input for numerical details (Exa: Fare, Passenger Class, Age)

	5). To exit the widget, simply press the cross button in the top right corner of the widget. 


Note: 

	1). The model is biased as we do not ask the user for all the features. We set some features as default, this results in algorithm making biased decision sometimes. Asking the user for all the input will be a little messy on the widget.So keeping all these scenarios in mind, the widget is developed in the way it is.

	2). The machine learning algorithm being used, is already tuned.
