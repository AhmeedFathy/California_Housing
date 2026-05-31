# California Housing Price Prediction

A machine learning project that predicts median house values in California districts using demographic, geographic, and housing features — including a live REST API deployment.

This project was built as part of my machine learning learning path and rebuilt in my own version to practice a complete end-to-end workflow: data exploration, preprocessing, model comparison, hyperparameter tuning, final evaluation, and API deployment.

## Live API Demo

The trained model is deployed as a REST API using FastAPI. Send house features → get a predicted price back in milliseconds.

**Request:**
```json
{
  "longitude": -122.23,
  "latitude": 37.88,
  "housing_median_age": 41.0,
  "total_rooms": 880,
  "total_bedrooms": 129,
  "population": 322,
  "households": 126,
  "median_income": 8.3252,
  "ocean_proximity": "NEAR BAY"
}
```
**Response:**
```json
{ "predicted_price_usd": 442926.1 }
```

## Project Goal

Predict `median_house_value` using features: longitude, latitude, housing median age, total rooms, total bedrooms, population, households, median income, ocean proximity.

## Workflow

1. **Data Exploration** — data types, missing values, distributions, correlations, geographic patterns
2. **Stratified Train/Test Split** — based on `median_income` categories to preserve distribution
3. **Preprocessing** — median imputation + standard scaling for numerical; most frequent imputation + one-hot encoding for `ocean_proximity`; combined with `ColumnTransformer`
4. **Model Training** — Linear Regression, Decision Tree Regressor, Random Forest Regressor
5. **Model Comparison** — 10-fold cross-validation with RMSE
6. **Hyperparameter Tuning** — GridSearchCV on Random Forest (27 combinations)
7. **Final Evaluation** — evaluated once on unseen test set
8. **API Deployment** — saved pipeline + model, wrapped in FastAPI, served via REST endpoint

## Model Performance

| Model | Mean RMSE | Std RMSE |
|---|---:|---:|
| Random Forest Regressor | 49,598.73 | 1,208.41 |
| Linear Regression | 64,906.39 | 2,148.86 |
| Decision Tree Regressor | 69,490.53 | 781.52 |

**Best Hyperparameters:** `n_estimators: 200, max_depth: None, min_samples_split: 2`

**Final Test Results:** RMSE: 47,073 | MAE: 30,864 | R²: 0.83

## API Usage

```bash
pip install fastapi uvicorn
python -m uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` to test interactively.

```python
import requests
response = requests.post("http://127.0.0.1:8000/predict", json={
    "longitude": -122.23, "latitude": 37.88, "housing_median_age": 41.0,
    "total_rooms": 880, "total_bedrooms": 129, "population": 322,
    "households": 126, "median_income": 8.3252, "ocean_proximity": "NEAR BAY"
})
print(response.json())
# → {"predicted_price_usd": 442926.1}
```

## Project Structure

California-Housing/
├── California_Housing.ipynb   # Full ML workflow
├── main.py                    # FastAPI deployment
├── housing.csv                # Dataset
└── README.md

## Key Takeaways

- `median_income` was the strongest predictor of house value
- Random Forest outperformed Linear Regression and Decision Tree on cross-validation
- Training metrics alone were misleading — cross-validation gave a fair comparison
- The model is deployed as a live REST API, not just a notebook experiment

## Tools

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · FastAPI · Uvicorn

## Author

**Ahmed Fathy** —  [GitHub](https://github.com/AhmeedFathy)
