# 💉 Insulin Tracking & Prediction System

An AI-powered diabetes management application that helps users track insulin injections, predict future insulin dosage requirements, and receive timely reminders. The system combines machine learning models with an interactive dashboard to support safer and more efficient insulin management.

## 🚀 Features

* 💉 Insulin Injection Tracking
* ⏰ Smart Reminder System
* 🤖 AI-Based Dosage Prediction
* 📈 Future Insulin Requirement Forecasting
* 📊 Interactive Data Visualization Dashboard
* 👤 User Registration & Login
* 🔔 Injection Due-Time Notifications
* 📅 Historical Injection Records

## 🧠 Machine Learning

The project utilizes **XGBoost Regression Models** to:

* Predict the required insulin bolus dosage
* Estimate the time gap until the next injection
* Generate future insulin schedules using recursive forecasting

### Model Performance

* 📉 Bolus Prediction MAE: **0.055 Units**
* ⏱️ Time Gap Prediction MAE: **2.56 Hours**

## 🏗️ System Workflow

1. User logs into the application.
2. Current glucose information is entered.
3. Machine learning models predict:

   * Recommended insulin dosage
   * Next injection timing
4. The system stores injection history.
5. Future insulin schedules are generated automatically.
6. Users receive reminders for upcoming injections.

## 📊 Key Features

### 👤 User Module

* Registration & Login
* Injection History Tracking
* Personalized Recommendations

### 🤖 Prediction Module

* Dosage Prediction
* Injection Time Prediction
* Recursive Future Forecasting

### 📈 Analytics Module

* Injection Trends
* Dosage Visualization
* Historical Data Analysis

## 🛠️ Tech Stack

* Python
* Streamlit
* XGBoost
* Pandas
* NumPy
* Plotly
* Scikit-Learn

## 🎯 Objectives

* Improve insulin management for diabetic patients
* Reduce risks of overdose and underdosage
* Provide intelligent insulin scheduling
* Support better diabetes monitoring through AI


