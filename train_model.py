from sklearn.preprocessing import StandardScaler, PowerTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import joblib
import os

def scale_frame(frame):
    df = frame.copy()
    X, y = df.drop(columns=['Price(euro)']), df['Price(euro)']
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scale = scaler.fit_transform(X.values)
    Y_scale = power_trans.fit_transform(y.values.reshape(-1, 1))
    return X_scale, Y_scale, power_trans, scaler  # ✅ ДОБАВЛЕНО scaler

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":  # ✅ ИСПРАВЛЕНО
    if not os.path.exists("./df_clear.csv"):
        raise FileNotFoundError("Файл df_clear.csv не найден")

    df = pd.read_csv("./df_clear.csv")
    X, Y, power_trans, scaler = scale_frame(df)  # ✅ ДОБАВЛЕНО scaler
    
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, random_state=42)
    
    params = {
        'alpha': [0.0001, 0.001, 0.01],
        'l1_ratio': [0.01, 0.05, 0.2],
        'penalty': ["l1", "l2", "elasticnet"],
        'loss': ['squared_error', 'huber'],
        'fit_intercept': [True, False],
    }

    mlflow.set_experiment("linear_model_cars")  # ✅ УБРАНЫ ПРОБЕЛЫ
    
    with mlflow.start_run():
        lr = SGDRegressor(random_state=42)
        clf = GridSearchCV(lr, params, cv=3, n_jobs=4)
        clf.fit(X_train, y_train.reshape(-1))
        
        best = clf.best_estimator_  # ✅ УБРАН ПРОБЕЛ
        
        y_pred = best.predict(X_val)
        y_price_pred = power_trans.inverse_transform(y_pred.reshape(-1, 1))  # ✅ УБРАН ПРОБЕЛ
        y_val_real = power_trans.inverse_transform(y_val)
        
        (rmse, mae, r2) = eval_metrics(y_val_real, y_price_pred)
        
        mlflow.log_param("alpha", best.alpha)  # ✅ УБРАНЫ ПРОБЕЛЫ
        mlflow.log_param("l1_ratio", best.l1_ratio)
        mlflow.log_param("penalty", best.penalty)
        mlflow.log_param("loss", best.loss)
        mlflow.log_param("fit_intercept", best.fit_intercept)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        
        mlflow.sklearn.log_model(best, "model", signature=signature)  # ✅ УБРАН ПРОБЕЛ
        
        with open("lr_cars.pkl", "wb") as file:
            joblib.dump(best, file)  # ✅ СОХРАНЯЕМ best, НЕ lr!

    dfruns = mlflow.search_runs()
    path2model = dfruns.sort_values("metrics.r2", ascending=False).iloc[0]['artifact_uri'].replace("file://", "") + '/model'
    
    with open("best_model.txt", "w") as f:  # ✅ ДОБАВЛЕНА ЗАПИСЬ ПУТИ!
        f.write(path2model)
        
    print(f"Model saved to: {path2model}")
