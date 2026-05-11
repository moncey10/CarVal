# 🚗 CarPrice AI — Car Price Prediction System

> An end-to-end Machine Learning web application that predicts used car prices based on key vehicle features using Random Forest Regression.


---

## 📌 Project Overview

CarPrice AI predicts the market price of a used car based on inputs like brand, year, fuel type, transmission, and kilometers driven. Built with a Random Forest model and served via a FastAPI REST API with a dark-themed frontend.

---

## 🚀 Features

- ✅ **ML Model** — Random Forest Regressor for accurate price prediction
- ✅ **FastAPI Backend** — REST API with `/predict` endpoint
- ✅ **Interactive Frontend** — Dark-themed UI for real-time price estimation
- ✅ **Feature Engineering** — Label encoding, outlier handling, feature selection
- ✅ **sklearn Pipeline** — Preprocessing + model in a single pipeline

---

## 🏗️ Project Structure

```
car-price-prediction/
├── app.py                  # FastAPI backend
├── train.py        # Data cleaning + model training
├── pipeline/
│   ├── __init__.py
│   └── build_pipeline.py   # sklearn Pipeline builder
├── index.html              # Frontend UI
├── artifacts/
│   └── model.pkl           # Trained model (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/car-price-prediction.git
cd car-price-prediction
```

### 2. Create virtual environment
```bash
python -m venv env
# Windows
env\Scripts\activate
# Mac/Linux
source env/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python preprocessing.py
```

### 5. Start FastAPI server
```bash
uvicorn app:app --reload
```

### 6. Open frontend
Open `index.html` in your browser.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Algorithm | Random Forest Regressor |
| R² Score | ~0.92 |
| Features Used | Brand, Year, KMs Driven, Fuel Type, Transmission, Owner |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Predict car price |

### Sample Request
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "Maruti",
    "year": 2019,
    "km_driven": 45000,
    "fuel": "Petrol",
    "transmission": "Manual",
    "owner": "First Owner"
  }'
```

### Sample Response
```json
{
  "predicted_price": 485000,
  "currency": "INR",
  "message": "Estimated market price"
}
```

---

## 🧠 How It Works

```
User enters car details
        ↓
FastAPI receives input
        ↓
Label Encoding for categorical features
        ↓
Random Forest predicts price
        ↓
Predicted price returned in INR
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | scikit-learn (Random Forest) |
| Backend | FastAPI + Uvicorn |
| Data Processing | Pandas, NumPy |
| Frontend | HTML, CSS, Vanilla JS |

---

## 👨‍💻 Author

**Moncey** — AI/ML Engineer  
Gujarat Technological University, 2026  
Internship: Scholar Clone (AI-powered EdTech)