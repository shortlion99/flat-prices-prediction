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

### 3) Install OpenMP runtime
```bash
brew install libomp
```

### 4) Add your API keys
Create a local `.env` from the `.env.example` template and fill in your own values
```bash
cp .env.example .env
```

### 5) Start the app

```bash
streamlit run app.py
```
