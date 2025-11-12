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

### 5) Add your models (Manual Step due to Github LFS size limit) 
- Download the model files (`sarimax_flattype_district.pkl` and `best_random_forest_model.pkl`) from the zipped folder.
- Place both files into the `models` folder within your local repository.


### 6) Start the app

```bash
streamlit run app.py
```
