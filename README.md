## 💧 Water Quality Prediction using Machine Learning

This repository contains a machine learning project aimed at predicting water quality based on various physicochemical attributes. The model is trained using a Random Forest classifier and includes steps for data preprocessing, imputation of missing values, feature scaling, and evaluation.

----


# 📂 Project Structure
php

Copy

Edit

water_quality_prediction/

├── app.py                    # Flask app to serve the model (optional for deployment) 

├── model.pkl               # Trained Random Forest model (if available)

├── scaler.pkl              # Saved StandardScaler object

├── dataset.csv             # Raw water quality dataset

├── templates/
│   └── index.html          # HTML page for user interface

├── static/
│   └── style.css           # Optional custom styling

├── README.md               # Project documentation

└── requirements.txt        # Python dependencies
---

# 📊 Dataset
The dataset contains multiple features that influence water quality, such as:

pH

Hardness

Solids

Chloramines

Sulfate

Conductivity

Organic Carbon

Trihalomethanes

Turbidity

Each row in the dataset includes these features and a binary label indicating potable (1) or not potable (0) water.

# 🛠️ Tools and Libraries Used
Python 🐍

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

# 📈 Model Building Steps

Load Dataset: Using Pandas from CSV.

Exploratory Data Analysis: Used Seaborn and Matplotlib to visualize missing values and feature distributions.

Missing Value Imputation: Used SimpleImputer with mean strategy.

Feature Scaling: Standardized using StandardScaler.

Train-Test Split: 80/20 split using train_test_split.

Model Training: Random Forest Classifier.

Evaluation Metrics: Accuracy Score, Classification Report.

# 🧪 Sample Code Snippet
python

Copy

Edit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

 Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

 Prediction & evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
----
# 🚀 Running the App
If you have a Flask app (app.py), you can run the web interface using:

bash

Copy

Edit

python app.py
Then open your browser and go to http://localhost:5000.

# 📦 Installation
bash
Copy
Edit
pip install -r requirements.txt
Or manually install dependencies:

bash

Copy

Edit

pip install pandas numpy matplotlib seaborn scikit-learn flask
# 🧠 Future Improvements
Add support for real-time sensor data input.

Explore deep learning models.

Improve UI using Bootstrap or React.

# 📜 License
This project is licensed under the MIT License.
