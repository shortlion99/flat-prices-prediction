# Flat Prices Prediction Model and Dashboard
## Aim
To develop a predictive analytics model and an interactive dashboard that enables the public to better understand, explore, and forecast property prices in Singapore. 

## Local Set Up

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv && source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```
```bash
brew bundle
```

### 3) Provide our trained model

- Place a serialized scikit-learn model at `models/model.pkl`.

### 4) Download the data
```bash
python data/download_data.py
```

### 5) Start the app

```bash
streamlit run app.py
```
