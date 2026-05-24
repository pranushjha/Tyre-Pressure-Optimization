# 🚚 AI Truck Fleet Tire Pressure Optimization

An AI-powered system that predicts optimal tire pressure and detects potential tire failure risks for truck fleets using Machine Learning models. This application helps fleet managers optimize tire maintenance, improve safety, and reduce operational costs.

## 🎯 Overview

This project combines Machine Learning with real-time tire monitoring to provide intelligent recommendations for truck fleet tire management. The system analyzes various factors including road conditions, vehicle load, climate, and tire age to predict optimal tire pressure and assess failure risks.

### Key Features

- **🤖 ML-Based Predictions**: Random Forest models for tire type classification and pressure optimization
- **⚠️ Failure Risk Detection**: Identifies high, medium, and low-risk tire conditions
- **🛣️ Multi-Road Support**: Handles highway, urban, and mountain road conditions
- **🌡️ Climate Adaptation**: Considers dry, rainy, and cold weather conditions
- **📊 Real-time Dashboard**: Interactive web interface for predictions
- **💡 Safety Recommendations**: Provides actionable advice based on predictions

## 📊 Why Tire Optimization Matters

- **Fleet Safety**: Incorrect tire pressure is a leading cause of truck accidents
- **Fuel Efficiency**: Optimized pressure can improve fuel efficiency by up to 5%
- **Cost Reduction**: Prevents early tire damage and reduces maintenance costs
- **Compliance**: Helps maintain regulatory standards for vehicle safety

## 🏗️ Project Architecture
├── app.py # Flask backend with ML models ├── index.html # Frontend UI ├── script.js # Frontend logic ├── styles.css # Styling ├── tire_test_data_large.csv # Training data for tire classification ├── fleet_tire_dataset_real.csv # Training data for pressure prediction └── README.md # Documentation

Code

## 📋 Technical Stack

- **Backend**: Python, Flask, Flask-CORS
- **Machine Learning**: scikit-learn (Random Forest)
- **Data Processing**: pandas, numpy
- **Frontend**: HTML5, CSS3, JavaScript, Tailwind CSS
- **API**: RESTful JSON endpoints

## 🗂️ Dataset Description

### tire_test_data_large.csv
Training data for tire type classification (Steer, Drive, Trailer)
- **Features**: roadtype, loadkg, axles, climate
- **Target**: steertire, drivetire, trailertire

### fleet_tire_dataset_real.csv
Real-world tire performance data for pressure prediction
- **Features**: 
  - roadtype, loadkg, axles, climate
  - temperature, avg_speed, tire_age, wear_level
  - load_per_axle (calculated)
- **Target**: optimal_psi
- **Risk Indicator**: failure_flag

## 🔧 Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/pranushjha/Tyre-Pressure-Optimization.git
cd Tyre-Pressure-Optimization
Install dependencies
bash
pip install -r requirements.txt
Verify data files Ensure both CSV files are present in the project root:
tire_test_data_large.csv
fleet_tire_dataset_real.csv
🚀 Usage
Start the Backend Server
bash
python app.py
The server will run on http://127.0.0.1:5000

Access the Web Interface
Open your browser and navigate to:

Code
http://localhost:5000
Making Predictions
Fill in the truck details:

Road Type: Highway, Urban, or Mountain
Load: Total load in kg
Axles: Number of axles (2-6)
Climate: Dry, Rainy, or Cold
Temperature: Current temperature in °C
Speed: Average speed in km/h
Tire Age: Age in years
Wear Level: Low, Medium, or High
Click "Predict Tire Optimization"

View the results:

Recommended tire types (Steer, Drive, Trailer)
Optimal tire pressure (PSI)
Failure risk assessment
Load per axle
Safety recommendations
📡 API Endpoints
POST /predict
Request:

JSON
{
  "roadType": "highway",
  "loadKg": 20000,
  "axles": 3,
  "climate": "dry",
  "temperature": 25,
  "speed": 80,
  "tireAge": 2,
  "wearLevel": "medium"
}
Response:

JSON
{
  "steerTire": "AllSeason",
  "driveTire": "HighPerformance",
  "trailerTire": "Standard",
  "pressure": 110.5,
  "failureRisk": "LOW",
  "loadPerAxle": 6666.67,
  "safetyAdvice": "Tire condition safe"
}
🤖 Machine Learning Models
Tire Type Classification
Model: Random Forest Classifier
Input Features: roadtype, loadkg, axles, climate
Output: steertire, drivetire, trailertire classifications
Models Used: 3 separate classifiers for each tire position
Pressure Prediction
Model: Random Forest Regressor
Input Features: 9 features including road type, load, climate, temperature, speed, tire age, wear level, and load per axle
Output: Optimal tire pressure (PSI)
Risk Assessment Logic
Code
if wear_level == "high" OR tire_age > 3 years:
    failure_risk = "HIGH"
    advice = "Replace tire immediately"
elif predicted_pressure > 115 PSI:
    failure_risk = "MEDIUM"
    advice = "Check tire pressure soon"
else:
    failure_risk = "LOW"
    advice = "Tire condition safe"
📈 Data Processing
Label Encoding: Categorical features (roadtype, climate, wear_level) are encoded
Feature Scaling: Load per axle is calculated from total load and number of axles
Model Training: Separate models trained for tire classification and pressure prediction
🎨 Frontend Features
Responsive design with Tailwind CSS
Real-time form validation
Interactive result dashboard
Safety statistics and information cards
Mobile-friendly interface
📊 Safety Statistics
The application highlights important road safety data:

1,264 road accidents occur daily in India
Nearly 426 deaths happen daily due to road accidents
Trucks contribute significantly to highway accidents due to tire failures
Improper tire pressure is a major cause of breakdowns
🔄 CORS Configuration
The application uses Flask-CORS to enable cross-origin requests, allowing the frontend to communicate with the backend from different origins.

📝 Code Structure
app.py
Data loading and preprocessing
Model training (Steer, Drive, Trailer classifiers + Pressure regressor)
Flask routes for static files and predictions
Prediction logic and risk assessment
script.js
Form submission handling
API communication with backend
Result display and UI updates
Error handling
index.html
Responsive layout structure
Form inputs for truck details
Result dashboard cards
Information sections
Safety statistics footer
🚀 Future Enhancements
 Real-time GPS integration
 Historical prediction tracking
 Fleet analytics dashboard
 Automated alerts for high-risk conditions
 Multiple language support
 Mobile app version
 IoT sensor integration
 Predictive maintenance scheduling
 Cost analysis reports
 Driver feedback integration
🐛 Error Handling
Input validation for all form fields
Error messages displayed to users
Console logging for debugging
Exception handling in API calls
📄 License
This project is open source and available under the MIT License.

👤 Author
Pranush Jha - GitHub Profile

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check out the issues page.

📞 Support
For support, open an issue on the GitHub repository or contact the maintainer.

Note: This application is designed for truck fleet management. For production deployment, ensure proper data validation, security measures, and compliance with local regulations.

Code

