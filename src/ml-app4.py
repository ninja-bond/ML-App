#for dataset
import sys
import pandas as pd
import numpy as np

#for app
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QComboBox, QTextEdit, QDoubleSpinBox, QSpinBox
import joblib

# sklearn metrics and model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

#for NN
import torch
from torch import nn, optim

#Random Forest
from sklearn.ensemble import RandomForestClassifier

#Linear Regression
from sklearn.linear_model import LinearRegression

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


#for the plot
import matplotlib 
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QT_VERSION_STR
matplotlib.use('QtAgg')  # Ensure Matplotlib uses the correct backend
print(f"Using PyQt6 version: {QT_VERSION_STR}")  # Debugging output
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

#Encoding
from sklearn.preprocessing import LabelEncoder

#catBoost - DT
import catboost
from catboost import CatBoostClassifier, Pool

#XGBoost
from xgboost import XGBClassifier, XGBRegressor, plot_importance

#LightGBM
from lightgbm import LGBMClassifier

#ensemble
from sklearn.ensemble import GradientBoostingClassifier

#naive bayes
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

#svm
from sklearn.svm import SVC

#knn
from sklearn.neighbors import KNeighborsClassifier

#for 3d plot 
# import plotly.express as px
# import plotly.graph_object as go

#analysis
from scipy.stats import chi2_contingency, f_oneway, ttest_ind
from sklearn.feature_selection import mutual_info_classif

#for plotting feature importance
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

#for websockets
import asyncio
import websockets
from PyQt6.QtCore import QThread, pyqtSignal



class WebSocketHandler(QThread):
    receivedData = pyqtSignal(str) #Signal to update UI
    
    def run(self):
        asyncio.run(self.connectWebSocket())
        
    async def connectWebSocket(self):
        uri = 'ws://localhost:8000/ws'
        async with websockets.connect(uri) as websocket:
            self.websocket = websocket
            while True:
                data = await websocket.recv()
                self.receivedData.emit(data)
                
    async def sendUpdate(self, message):
        if hasattr(self, 'websocket'):
            await self.websocket.send(message)



class MLApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.dataset = None
        self.model = None
        self.canvas = None
        self.websocket = WebSocketHandler()
        self.websocket.receivedData.connect(self.updateFromWebSocket)
        self.websocket.start()
    
    def initUI(self):
        layout = QVBoxLayout()
        
        self.upload_btn = QPushButton('Upload Dataset')
        self.upload_btn.clicked.connect(self.loadDataset)
        layout.addWidget(self.upload_btn)
        
        self.cleaning_label = QLabel('Select Data Imputation Method:')
        layout.addWidget(self.cleaning_label)
        
        self.cleaning_method = QComboBox()
        self.cleaning_method.addItems([
            'No Cleaning', 'Mean Imputation', 'Median Imputation', 'Mode Imputation', 'KNN Imputation',
            'Multiple Imputation (MICE)', 'Regression Imputation', 'Hot Deck Imputation'
        ])
        layout.addWidget(self.cleaning_method)
        
        self.relation_btn = QPushButton('Analyze Column Relationships')
        self.relation_btn.clicked.connect(self.analyzeRelations)
        layout.addWidget(self.relation_btn)
        
        self.label = QLabel('Select Algorithm:')
        layout.addWidget(self.label)
        
        self.model_selection = QComboBox()
        self.model_selection.addItems(['Linear Regression', 'LR(perceptron-trick)', 'Logistic Regression(Binary Classification)', 'Logistic Regression(Multi-Class Classification)', 'SVM(binary classification)', 'Naive Bayes(GaussianNB)', 'Naive Bayes(MultinomialNB)', 'Naive Bayes(BernoulliNB)', 'KNN', 'Decision Tree(C4.5)', 'Decision Tree(ID3)', 'Decision Tree(CART)', 'Random Forest', 'XGBoost(Classification)', 'XGBoost(Regression)', 'catboost', 'LightGBM', 'Neural Network (DL)', 'Gradient Boosting'])
        layout.addWidget(self.model_selection)
        
        
        
        
        #-----------svm-----------
        
        self.kernel_label = QLabel("Kernel:")
        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems(["linear", "rbf", "poly", "sigmoid"])
        # self.kernel_combo.setValue("linear")

        self.c_label = QLabel("C (Regularization):")
        self.c_input = QDoubleSpinBox()
        self.c_input.setRange(0.01, 100.0)
        self.c_input.setSingleStep(0.1)
        self.c_input.setValue(1.0)

        # # Initially hide them
        self.kernel_label.hide()
        self.kernel_combo.hide()
        self.c_label.hide()
        self.c_input.hide()

        layout.addWidget(self.kernel_label)
        layout.addWidget(self.kernel_combo)
        layout.addWidget(self.c_label)
        layout.addWidget(self.c_input)

        # Connect model selection change event
        self.model_selection.currentTextChanged.connect(self.toggle_params)


        #-----------svm-----------
        
        #--------------knn------------------
        # KNN Hyperparameter Input
        self.k_label = QLabel("Number of Neighbors (k):")
        self.k_input = QSpinBox()
        self.k_input.setRange(1, 50)  # Set a reasonable range for k
        self.k_input.setValue(5)  # Default value

        # Initially hide them
        self.k_label.hide()
        self.k_input.hide()

        layout.addWidget(self.k_label)
        layout.addWidget(self.k_input)

        # Connect model selection change event
        self.model_selection.currentTextChanged.connect(self.toggle_params)
        
        #--------------knn------------------

        
        
        self.train_btn = QPushButton('Train Model')
        self.train_btn.clicked.connect(self.trainModel)
        layout.addWidget(self.train_btn)
        
        
        #for feature importance
        # self.model_selection.currentTextChanged.connect(self.showFeatureImportance)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)
        
        self.save_btn = QPushButton('Save Model')
        self.save_btn.clicked.connect(self.saveModel)
        self.save_btn.setEnabled(False)
        layout.addWidget(self.save_btn)
        
        # Background Color Selection
        self.color_dropdown = QComboBox()
        self.color_dropdown.addItems(["lightgray", "white", "blue", "green", "yellow", "pink", "black"])
        layout.addWidget(self.color_dropdown)

        # Change Background Button
        self.change_color_button = QPushButton("Change Background Color")
        self.change_color_button.clicked.connect(self.change_background)
        layout.addWidget(self.change_color_button)

        
        # Placeholder for the plot
        self.plot_layout = QVBoxLayout()
        layout.addLayout(self.plot_layout)
        
        #for feature importance plots 
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        
        # self.plot_feature_importance()

        
    
    def loadDataset(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'CSV Files (*.csv)')
        if file_path:
            self.dataset = pd.read_csv(file_path)
            self.output_text.append(f'Dataset loaded: {file_path}')
            self.output_text.append(f'Shape: {self.dataset.shape}')
            
    def applyDataCleaning(self, X):
        """ Apply selected data cleaning method """
        cleaning_choice = self.cleaning_method.currentText()
        
        if cleaning_choice == "Mean Imputation":
            imputer = SimpleImputer(strategy="mean")
            X_cleaned = imputer.fit_transform(X)
            self.output_text.append("Applied Mean Imputation.")
        
        elif cleaning_choice == "Median Imputation":
            imputer = SimpleImputer(strategy="median")
            X_cleaned = imputer.fit_transform(X)
            self.output_text.append("Applied Median Imputation.")
        
        elif cleaning_choice == "Mode Imputation":
            imputer = SimpleImputer(strategy="most_frequent")
            X_cleaned = imputer.fit_transform(X)
            self.output_text.append("Applied Mode Imputation.")
        
        elif cleaning_choice == "KNN Imputation":
            imputer = KNNImputer(n_neighbors=5)
            X_cleaned = imputer.fit_transform(X)
            self.output_text.append("Applied KNN Imputation.")
        
        elif cleaning_choice == "Multiple Imputation (MICE)":
            imputer = IterativeImputer(max_iter=10, random_state=42)
            X_cleaned = imputer.fit_transform(X)
            self.output_text.append("Applied Multiple Imputation (MICE).")
        
        elif cleaning_choice == "Regression Imputation":
            imputer = IterativeImputer(estimator=RandomForestClassifier(), max_iter=10, random_state=42)
            X_cleaned = imputer.fit_transform(X)
            self.output_text.append("Applied Regression Imputation using Random Forest.")
        
        elif cleaning_choice == "Hot Deck Imputation":
            X_cleaned = X.copy()
            for col in X.columns:
                missing_mask = X_cleaned[col].isnull()
                non_missing_values = X_cleaned[col].dropna().values
                if len(non_missing_values) > 0:
                    X_cleaned.loc[missing_mask, col] = np.random.choice(non_missing_values, size=missing_mask.sum(), replace=True)
            self.output_text.append("Applied Hot Deck Imputation.")
        
        else:  # No Cleaning
            X_cleaned = X.values
            self.output_text.append("No Data Cleaning applied.")
        
        return pd.DataFrame(X_cleaned, columns=X.columns)
    
    def toggle_params(self, text):
        
        if text == "SVM(binary classification)":
            self.kernel_label.show()
            self.kernel_combo.show()
            self.c_label.show()
            self.c_input.show()
        elif text == "KNN":
            self.k_label.show()
            self.k_input.show()
        else:
            self.kernel_label.hide()
            self.kernel_combo.hide()
            self.c_label.hide()
            self.c_input.hide()

    
    def trainModel(self):
        if self.dataset is None:
            self.output_text.append('Please upload a dataset first!')
            return
        
        X = self.dataset.iloc[:, :-1]
        y = self.dataset.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        
        model_choice = self.model_selection.currentText()
        
        if model_choice == 'Linear Regression':
            self.model = LinearRegression()
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            acc1 = mean_absolute_error(y_test, y_pred)
            acc2 = mean_squared_error(y_test, y_pred)
            acc3 = r2_score(y_test, y_pred)
            
            #plot LR
            self.plotResults(y_test, y_pred, 'Linear Regression')

            self.output_text.append(f'Linear Regression (Mean Absolute Error): {acc1: .4f}')
            self.output_text.append(f'Linear Regression (Mean Squared Error): {acc2: .4f}')
            self.output_text.append(f'Linear Regression (r2 score): {acc3: .4f}')
            
            #send updates to websocket
            asyncio.run(self.websocket.sendUpdate(str(r2_score)))

            
        elif model_choice == 'LR(perceptron trick(self))':
            X0 = np.insert(X_train, 0, 1, axis = 1)
            weights = np.ones(X0.shape[1])
            lr = 0.1
            
            for i in range(100):
                j = np.randint(0, 100)
                y_hat = step(np.dot(X0[j], weights)) #step returns 0 or 1 based on threshold 0
                weights = weights + lr*(y_train[j]-y_hat)
                
            # return weights[0], weights[1:]
            
        elif model_choice == 'Logistic Regression(Binary Classification)':
            self.model = LogisticRegression()
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            self.output_text.append(f'Accuracy: {acc:.4f}')
            self.output_text.append(f'Confusion Matrix: {confusion_matrix(y_test, y_pred)}')
            
        elif model_choice == 'Logistic Regression(Multi-Class Classification)':
            self.model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            acc = classification_report(y_test, y_pred)
            self.output_text.append(classification_report(y_test, y_pred))
        
        elif model_choice == 'svm(binary classification)':
            kernel_type = self.kernel_combo.currentText()
            c_value = self.c_input.value()

            self.model = SVC(kernel=kernel_type, C=c_value)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            self.output_text.append(f'SVM (Kernel: {kernel_type}, C: {c_value}) Accuracy: {acc:.4f}')

        elif model_choice == 'KNN':
            k_value = self.k_input.value()

            self.model = KNeighborsClassifier(n_neighbors=k_value)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            self.output_text.append(f'KNN (k={k_value}) Accuracy: {acc:.4f}')

        elif model_choice == 'Naive Bayes(BernoulliNB)':   
            le = LabelEncoder()
            # print(y_test[0])
            y_train = le.fit_transform(y_train)
            y_test= le.fit_transform(y_test)
            print(y_test[0])
            
            self.model = BernoulliNB()
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            self.output_text.append(f'BernoulliNB Accuracy: {acc:.4f}')
        
        elif model_choice == 'Naive Bayes(MultinomialNB)':
            print(y_test)
            self.model = MultinomialNB()
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            print(y_pred)
            acc = accuracy_score(y_test, y_pred)
            self.output_text.append(f'MultinomialNB Accuracy:{acc:.4f}')
        
        elif model_choice == 'Naive Bayes(GaussianNB)':
            self.model = GaussianNB()
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            self.output_text.predict(f'GaussianNB Accuracy:{acc:.4f}')
            
        elif model_choice == 'catboost':
            self.model = CatBoostClassifier(iterations=200, depth = 6, learning_rate=0.1, verbose=0)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            self.output_text.append(f'CatBoost Accuracy:{acc:.4f}')
            
        elif model_choice == 'LightGBM':
            self.model = LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=6)
            self.model.fit(X_train, y_train)
            y_pred=self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            self.output_text.append(f'LightGBM Accuracy: {acc:.4}')
            
            #getting feature importance
            feature_importances = self.model.feature_importances_
            self.plotFeatureImportance(feature_importances, data.feature_names)
            
                    
        elif model_choice == 'XGBoost(Regression)':
            # le = LabelEncoder()
            # y_train_encoded = le.fit_transform(y_train)
            # y_test_encoded = le.transform(y_test)
            
            label_encoders = {}
            for col in X.select_dtypes(include=["object"]).columns:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col])
                X_test[col] = le.transform(X_test[col])
                label_encoders[col] = le
            # if y_train.dtype == 'float64' or y_train.dtype == 'int64':
            #     le = LabelEncoder()
            y_train = y_train.astype(float)
            y_test = y_test.astype(float)
            y_train = y_train.fillna(y_train.median())
            y_test = y_test.fillna(y_test.median())
            
            self.model = XGBRegressor(objective = 'reg:squarederror', n_estimators=200, learning_rate=0.1, max_depth=6)
            self.model.fit(X_train, y_train, eval_set=[(X_train, y_train)])
            y_pred = self.model.predict(X_test)
            
            #regression metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            self.output_text.append(f'XGBoost MSE: {mse:.4f}')
            self.output_text.append(f'XGBoost R2 score:{r2:.4f}')
            
            #getting feature importance
            feature_importances = self.model.feature_importances_
            self.output_text.append(f"Feature Importances: {feature_importances.tolist()}")
            # self.(self.model)
            
            asyncio.run(self.websocket.sendUpdate(str(feature_importances)))

            
            plt.figure(figsize=(10, 6))
            plot_importance(self.model)
            plt.show()
            
            

            #self.plotFeatureImportance(feature_importances, self.dataset.feature_names)
            
            #self.plotFeatureImportance(feature_importances, self.dataset.columns[:-1])  # Exclude target column
        
        elif model_choice == 'XGBoost(Classification)':
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)
            self.model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, use_label_encoder=False, eval_metrics='logloss')
            self.model.fit(X_train, y_train_encoded, eval_set=[(X_test, y_test_encoded)])
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test_encoded, y_pred)
            self.output_text.append(f'XGBoost Accuracy: {acc:.4f}')
            
            #getting feature importance
            feature_importances = self.model.feature_importances_
            #self.plotFeatureImportance(feature_importances, self.dataset.feature_names)
            self.plotFeatureImportance(feature_importances, self.dataset.columns[:-1])  # Exclude target column
            
        elif model_choice == 'Gradient Boosting':
            if y.dtype == 'object':
                le=LabelEncoder()
                y_train=le.fit_transform(y_train)
                y_test=le.fit_transform(y_test)
            self.model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=6)
            self.model.fit(X_train, y_train)
            y_pred=self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            self.output_text.append(f'Gradient Boosting Accuracy: {acc:.4f}')
            
        elif model_choice == 'Decision Tree':
            self.model = DecisionTreeClassifier()
            self.model.fit(X, y)

            importances = clf.feature_importances_
            print(importances)
            
            
        elif model_choice == 'Random Forest (ML)':
            self.model = RandomForestClassifier(n_estimators=100)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            self.output_text.append(f'Random Forest Accuracy: {acc:.4f}')
            
            #getting feature importance
            feature_importances = self.model.feature_importances_
            self.plotFeatureImportance(feature_importances, data.feature_names)

        elif model_choice == 'Neural Network (DL)':
            class SimpleNN(nn.Module):
                def __init__(self, input_size):
                    super(SimpleNN, self).__init__()
                    self.fc1 = nn.Linear(input_size, 16)
                    self.relu = nn.ReLU()
                    self.fc2 = nn.Linear(16, 2)  # Binary classification
                    
                def forward(self, x):
                    x = self.fc1(x)
                    x = self.relu(x)
                    x = self.fc2(x)
                    return x
            
            input_size = X_train.shape[1]
            self.model = SimpleNN(input_size)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.01)
            
            X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
            
            for epoch in range(10):  # Small training loop
                optimizer.zero_grad()
                outputs = self.model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
            
            self.output_text.append('Neural Network trained successfully!')
        
        
        self.save_btn.setEnabled(True)
        
        
    #def plotFeatureImportance(self, importances, feature_names):
        # self.ax.clear()
        # indices = np.argsort(importances)
        # self.ax.barh(range(len(indices)), importances[indices], align = 'center')
        # self.ax.set_yticks(range(len(indices)))
        # self.ax.set_yticklabels(np.array(feature_names)[indices])
        # self.ax.set_xlabel('Feature Importance Score')
        # self.ax.set_title('Feature Importance')
        
        
        # if self.canvas is not None:
        #     layout = self.plot_layout
        #     layout.removeWidget(self.canvas)
        #     self.canvas.deleteLater()
        #     self.canvas = None
            
        # self.canvas.draw()
        
        
        """Plot actual vs predicted values inside the PyQt app."""
        if self.canvas:
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
            self.canvas = None

        # Create Matplotlib figure
        # fig, ax = plt.subplots(figsize=(5, 3))
        # ax.scatter(range(len(y_test)), y_test, color='blue', label='Actual', alpha=0.6)
        # ax.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted', alpha=0.6)
        # ax.set_xlabel('Sample Index')
        # ax.set_ylabel('Class Label')
        # ax.set_title(title)
        # ax.legend()

        #     # Embed Matplotlib figure inside PyQt
        # self.canvas = FigureCanvas(fig)
        # self.plot_layout.addWidget(self.canvas)
        # self.canvas.draw()
        # self.plot_layout.update()
        
        # fig = plt.figure(figsize=(10, 6))
        # plot_importance(self.model)
        #plt.show()
        
    def plot_feature_importance(self):
        self.ax.clear()  # Clear previous plot
        plot_importance(self.model, ax=self.ax)  # Plot feature importance
        self.canvas.draw()  # Refresh canvas
        
    def plotResults(self, y_test, y_pred, title):
        """Plot actual vs predicted values inside the PyQt app."""
        if self.canvas:
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
            self.canvas = None

        # Create Matplotlib figure
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.scatter(range(len(y_test)), y_test, color='blue', label='Actual', alpha=0.6)
        ax.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted', alpha=0.6)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Class Label')
        ax.set_title(title)
        ax.legend()

        # Embed Matplotlib figure inside PyQt
        self.canvas = FigureCanvas(fig)
        self.plot_layout.addWidget(self.canvas)
        self.canvas.draw()
        self.plot_layout.update()
        
        
    def showFeatureImportance(self):
        selected_model = self.model_selection.currentText()
        if selected_model in ['Random Forest', 'XGBoost', 'Decision Trees']:
            self.output_text.append('Feature Importance:')
            feature_importance = self.model.feature_importances_
            for i, col in enumerate(self.dataset.columns[:-1]):
                self.output_text(f'{col}: {feature_importance[i]}')
    
    def analyzeRelations(self):
        """Analyze column relationships using different tests."""
        if self.dataset is None:
            self.output_text.append("Please upload a dataset first!")
            return
        
        self.output_text.append("\n Running Column Relationship Analysis...\n")

        # 1️ Correlation (for numerical columns) 
        # N Vs N
        corr_matrix = self.dataset.corr(meathod = 'pearson')
        self.output_text.append(" Pearson Correlation Matrix:\n" + corr_matrix.to_string() + "\n")

        # N Vs N
        corr_matrix = self.dataset.corr(method='spearman')
        self.output_text.append(" Spearman Correlation: \n" + corr_matrix.to_string() + "\n")
        
        # 2️ Chi-Square Test (for categorical variables)
        # C Vs C
        cat_columns = self.dataset.select_dtypes(include=['object']).columns
        if len(cat_columns) >= 2:
            col1, col2 = cat_columns[:2]
            contingency_table = pd.crosstab(self.dataset[col1], self.dataset[col2])
            chi2, p, _, _ = chi2_contingency(contingency_table)
            self.output_text.append(f"Chi-Square Test between '{col1}' and '{col2}': p-value = {p:.5f}\n")

        # 3️ ANOVA Test (comparing a categorical with a numerical column)
        # N Vs C
        if len(cat_columns) > 0:
            num_columns = self.dataset.select_dtypes(include=['int64', 'float64']).columns
            if len(num_columns) > 0:
                cat_col = cat_columns[0]
                num_col = num_columns[0]
                groups = [self.dataset[self.dataset[cat_col] == category][num_col] for category in self.dataset[cat_col].unique()]
                f_stat, p_value = f_oneway(*groups)
                self.output_text.append(f" ANOVA Test between '{cat_col}' and '{num_col}': p-value = {p_value:.5f}\n")


        # 4️ Mutual Information (for feature importance)
        if self.dataset.shape[1] > 1:
            X = self.dataset.iloc[:, :-1]
            y = self.dataset.iloc[:, -1]
            if y.dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y)

            mi_scores = mutual_info_classif(X, y)
            self.output_text.append(f" Mutual Information Scores: {mi_scores}\n")

        self.output_text.append("Relationship Analysis Completed!\n")
    
    def saveModel(self):
        # file_path, _ = QFileDialog.getSaveFileName(self, 'Save Model', '', 'Model Files (*.pkl)')
        # if file_path and self.model:
        #     joblib.dump(self.model, file_path)
        # #     self.output_text.append(f'Model saved to {file_path}')
        
        # file_path, _ = QFileDialog.getSaveFileName(self, 'Save Model', '', 'Model Files (*.pkl *.cbm)')
        # if file_path and self.model:
        #     if isinstance(self.model, CatBoostClassifier):
        #         self.model.save_model(file_path, format="cbm")
        #     else:
        #         joblib.dump(self.model, file_path)
        #     self.output_text.append(f'Model saved to {file_path}')
        
        file_path, _ = QFileDialog.getSaveFileName(self, 'Save Model', '', 'Model Files(*.pkl *.cbm *.txt)')
        if file_path and self.model:
            if isinstance(self.model, CatBoostClassifier):
                self.model.save_model(file_path, format='cbm')
            elif isinstance(self.model, LGBMClassifier):
                self.model.booster_.save_model(file_path)
            elif isinstance(self.model, XGBClassifier):
                self.model.save_model(file_path)
            else:
                joblib.dump(self.model, file_path)
            self.output.append(f'Model Saved to {file_path}')
            
    def change_background(self):
        
        selected_color = self.color_dropdown.currentText()
        self.upload_btn.setStyleSheet(f"background-color: {selected_color};")

    def updateFromWebSocket(self, message):
        self.output_text.append(f"[TEAM] {message}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MLApp()
    window.show()
    sys.exit(app.exec())
