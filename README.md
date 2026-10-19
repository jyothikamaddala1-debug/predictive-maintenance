# Predictive Maintenance System

An end-to-end machine learning project that predicts whether a machine is likely to fail based on operational and sensor data.

## Project Overview

This project uses supervised machine learning to predict machine failures before they occur.

The system:
- Loads and preprocesses machine sensor data
- Performs exploratory data analysis
- Trains multiple machine learning models
- Compares model performance using evaluation metrics
- Selects the best-performing model
- Provides predictions through a FastAPI REST API

## Dataset

Dataset: AI4I 2020 Predictive Maintenance Dataset

The dataset contains machine information such as:
- Machine type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

Target variable:
- `Machine failure`
  - `0` → No machine failure
  - `1` → Machine failure

## Machine Learning Models

The following machine learning algorithms were trained and evaluated:

1. Logistic Regression
2. Balanced Logistic Regression
3. Random Forest
4. Gradient Boosting
5. Support Vector Machine (SVM)

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 96.85% | 66.67% | 14.71% | 24.10% |
| Balanced Logistic Regression | 82.70% | 14.36% | 82.35% | 24.45% |
| Random Forest | 97.95% | 71.43% | 66.18% | 68.70% |
| Gradient Boosting | 98.70% | 92.00% | 67.65% | 77.97% |
| SVM | 91.70% | 27.93% | 91.18% | 42.76% |

### Best Model

Gradient Boosting achieved the best overall performance with:

- Accuracy: **98.70%**
- Precision: **92.00%**
- Recall: **67.65%**
- F1 Score: **77.97%**

Gradient Boosting was selected as the final model for deployment.

## FastAPI Deployment

The trained Gradient Boosting model is deployed using FastAPI.

### Run the API

Activate the virtual environment and run:

```bash
uvicorn api.main:app --reload

The API will be available at:
http://127.0.0.1:8000

Interactive API documentation:
http://127.0.0.1:8000/docs

Prediction Endpoint
POST /predict

Example request:
{
  "machine_type": "L",
  "air_temperature": 300,
  "process_temperature": 310,
  "rotational_speed": 1500,
  "torque": 40,
  "tool_wear": 100
}

Example response:
{
  "prediction": 0,
  "result": "No Machine Failure Predicted"
}

The API also validates the machine type and rejects invalid values.