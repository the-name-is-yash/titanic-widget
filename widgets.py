import sys


# Importing PyQT
from PyQt5.QtWidgets import QApplication, QLineEdit, QCheckBox, QMainWindow, QTabWidget, QMessageBox
from PyQt5.QtWidgets import QLabel, QPushButton
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QFont, QIntValidator

# import sklearn and knneighbours
from sklearn.neighbors import KNeighborsClassifier


import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# Importing Dataframe
import pandas as pd
df = pd.read_csv("train.csv")


# Class for matplotlib canvas
class Canvas(FigureCanvas):
    def __init__(self, parent):
        fig, ax = plt.subplots(2, figsize=(6,10))
        super().__init__(fig)
        self.setParent(parent)


        pd.crosstab(df["Sex"], df["Survived"]).plot.bar(figsize=(5,5), xlabel="Sex", ylabel = "Passenger Frequency", ax=ax[0]);


        pd.crosstab(df.Age[df["Survived"] == 1], df.Survived[df["Survived"] == 1]).plot(figsize=(5, 5), color="orange", ylabel="Passengers survived", ax=ax[1]);

        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=0)



class AppDemo(QWidget):

    def __init__(self):
        super().__init__()
        self.resize(900,900)

        chart = Canvas(self)


class MainWindow(QTabWidget):

    def __init__(self):
        super().__init__()

        print("Describing the dataset\n")

        print(df.describe())

        print("============================================================================================\n")

        print("Information about the dataset.\n")

        print(df.info())

        print("============================================================================================\n")




        self.initUI()

        # Adding the Description
        self.label = QLabel("Hello Passenger ! Please enter your details below.", self)
        self.label.setFont(QFont("Bold",20))

        # Adding the gender
        self.gender = QLabel("Gender:",self)
        self.gender.move(700,165)
        self.gender.setFont(QFont("Bold",10))

        # Add prediction label
        self.prediction = QLabel("Your Predictions will appear here!",self)
        self.prediction.move(650,600)
        self.prediction.setFont(QFont("Bold",15))

        # Moving the Description
        self.label.move(700,50)

        # Adding the Title
        self.setWindowTitle("Surviving the unsinkable")

        # Set the window size
        self.setFixedSize(QSize(1500,1000))

        # Adding the TextBox for age
        self.age = QLabel("Age: ", self)
        self.age.move(700,220)
        self.age.setFont(QFont("Bold", 10))
        self.age_no = QLineEdit(self)
        self.age_no.setValidator(QIntValidator(1,100,self))
        self.age_no.move(750,215)
        self.age_no.resize(150, 30)

        # Adding the TextBox for Fare
        self.fare = QLabel("Fare: ", self)
        self.fare.move(700,280)
        self.fare.setFont(QFont("Bold", 10))
        self.fare_no = QLineEdit(self)
        self.fare_no.setValidator(QIntValidator(1,21,self))
        self.fare_no.move(750,275)
        self.fare_no.resize(150, 30)

        # Adding the TextBox for pclass
        self.pclass = QLabel("PClass: ", self)
        self.pclass.move(700,340)
        self.pclass.setFont(QFont("Bold", 10))
        self.pclass_box = QLineEdit(self)
        self.pclass_box.setValidator(QIntValidator(1,3,self))
        self.pclass_box.move(770,335)
        self.pclass_box.resize(150, 30)

        # Adding the predict button
        self.button = QPushButton("Predict",self)
        self.button.setCheckable(True)
        self.button.move(745,400)

        # One time checking False, another True
        self.button.setCheckable(True)


        self.button.clicked.connect(self.button_clicked)


        # Adding another button
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.fare_no)

        self.display_2 = self.button_clicked()

        # Adding the tabs
        self.tab1 = QWidget(self)
        self.addTab(demo, "Prediction Machine")

    # Print message when button clicked and storing the variable
    def button_clicked(self):
        self.button_is_checked = self.button.isChecked()

        self.model_results = self.sklearn_model()

        self.print_age_no = self.age_no.text()
        self.print_fare_no = self.age_no.text()
        self.print_pclass = self.pclass_box.text()

        self.gender = self.uncheck(Qt.Checked)



        if self.print_age_no != "":
            # Setting other parameters as most recurring values as they are unimportant
            data = [[int(self.print_age_no), False, 0, True, 3, False, int(self.print_fare_no), False, 207, 0.892, 3, int(self.print_pclass), 2,0,320]]

            df_1 = pd.DataFrame(data, columns=['Age', 'Age_is_missing', 'Cabin', 'Cabin_is_missing', 'Embarked',
                                               'Embarked_is_missing', 'Fare', 'Fare_is_missing', 'Name', 'Parch',
                                               'PassengerId', 'Pclass', 'Sex', 'SibSp', 'Ticket'])

            model = self.model_results.predict(df_1)

            if model[0] == 1:
                self.prediction.setText("Congratulations! You Survived")

            else:
                self.prediction.setText("Unfortunately, You didn't survive")

    def sklearn_model(self):

        df = pd.read_csv("train.csv")

        temp = df.copy()

        X = temp.drop("Survived", axis=1)
        y = temp["Survived"]

        # Split the data into train & test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for label, content in X_train.items():
            if not pd.api.types.is_numeric_dtype(content):

                if pd.isna(content).sum():
                    # Add binary column to indicate if values were missing
                    X_train[label + "_is_missing"] = pd.isna(content)

                    # Fill the columns with categorical data
                    X_train[label] = pd.Categorical(content).codes + 1

                else:
                    # Fill the columns with categorical data
                    X_train[label] = pd.Categorical(content).codes + 1

        for label, content in X_test.items():
            if not pd.api.types.is_numeric_dtype(content):

                if pd.isna(content).sum():
                    # Add binary column to indicate if values were missing
                    X_test[label + "_is_missing"] = pd.isna(content)

                    # Fill the columns with categorical data
                    X_test[label] = pd.Categorical(content).codes + 1

                else:
                    # Fill the columns with categorical data
                    X_test[label] = pd.Categorical(content).codes + 1

        # Fill in the missing values of integer columns with median
        for label, content in X_train.items():
            if pd.api.types.is_numeric_dtype(content):

                if pd.isna(content).sum():
                    # Add binary column to indicate if values were missing
                    X_train[label + "_is_missing"] = pd.isna(content)

                    # Fill the columns with median
                    X_train[label] = content.fillna(content.median())

                else:
                    # Fill the columns with median
                    X_train[label] = content.fillna(content.median())

        for label, content in X_test.items():
            if pd.api.types.is_numeric_dtype(content):

                if pd.isna(content).sum():
                    # Add binary column to indicate if values were missing
                    X_test[label + "_is_missing"] = pd.isna(content)

                    # Fill the columns median
                    X_test[label] = content.fillna(content.median())

                else:
                    # Fill the columns with median
                    X_test[label] = content.fillna(content.median())

        X_test["Embarked_is_missing"] = X_test["Embarked"].isna()
        X_test['Fare_is_missing'] = X_test["Fare"].isna()
        X_train['Fare_is_missing'] = X_train["Fare"].isna()

        # Sorting the datasets as per alphabetical order
        X_test = X_test.sort_index(axis=1)
        X_train = X_train.sort_index(axis=1)

        # Training model
        ideal_model = KNeighborsClassifier(leaf_size = 1, n_neighbors = 19, p = 1)
        ideal_model.fit(X_train, y_train)

        model_score = ideal_model.score(X_test, y_test)

        return ideal_model


    def initUI(self):
            # creating check box
        self.male = QCheckBox("Male", self)

            # setting geometry
        self.male.setGeometry(770, 160, 100, 30)
        self.male.setFont(QFont("Bold", 10))

            # creating check box
        self.female = QCheckBox("Female", self)

            # setting geometry
        self.female.setGeometry(850, 160, 100, 30)
        self.female.setFont(QFont("Bold", 10))

            # calling the uncheck method if any check box state is changed
        self.male.stateChanged.connect(self.uncheck)
        self.female.stateChanged.connect(self.uncheck)

            # uncheck method
    def uncheck(self, state):

                # checking if state is checked
        if state == Qt.Checked:

                    # if first check box is selected
            if self.sender() == self.male:

                        # making other check box to uncheck
                self.female.setChecked(False)
                return 2

                    # if second check box is selected
            elif self.sender() == self.female:

                        # making other check box to uncheck
                self.male.setChecked(False)
                return 1

            self.show()

app = QApplication(sys.argv)
demo = AppDemo()
demo.show()
mw = MainWindow()
mw.show()
sys.exit(app.exec_())