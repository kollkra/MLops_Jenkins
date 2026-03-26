import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

print("Загрузка данных...")
df = pd.read_csv('df_clear.csv')

# Разделение на признаки и целевую переменную
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

print(f"Признаки: {X.columns.tolist()}")
print(f"Размер X: {X.shape}, Размер y: {y.shape}")

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Сохранение scaler
joblib.dump(scaler, 'scaler.pkl')
print("Scaler сохранен в scaler.pkl")

# Настройка MLflow
mlflow.set_experiment("California_Housing_Model")

# Параметры для GridSearchCV
param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'max_iter': [1000, 2000],
    'tol': [1e-3, 1e-4]
}

print("\nНачало обучения модели с GridSearchCV...")
sgd = SGDRegressor(random_state=42)

grid_search = GridSearchCV(
    sgd, 
    param_grid, 
    cv=5, 
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

with mlflow.start_run():
    # Обучение модели
    grid_search.fit(X_train_scaled, y_train)
    
    # Лучшая модель
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"\nЛучшие параметры: {best_params}")
    
    # Предсказания
    y_pred_train = best_model.predict(X_train_scaled)
    y_pred_test = best_model.predict(X_test_scaled)
    
    # Метрики качества
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\nМетрики на тренировочной выборке:")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  MAE: {train_mae:.2f}")
    print(f"  R²: {train_r2:.4f}")
    
    print(f"\nМетрики на тестовой выборке:")
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  MAE: {test_mae:.2f}")
    print(f"  R²: {test_r2:.4f}")
    
    # Логирование в MLflow
    mlflow.log_params(best_params)
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("test_rmse", test_rmse)
    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("train_r2", train_r2)
    mlflow.log_metric("test_r2", test_r2)
    
    # Логирование модели
    mlflow.sklearn.log_model(best_model, "model")
    
    # Получение URI модели
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"
    
    print(f"\nМодель залогирована в MLflow")
    print(f"URI модели: {model_uri}")
    
    # Сохранение пути к модели в файл
    with open('best_model.txt', 'w') as f:
        f.write(model_uri)
    
    print(f"Путь к модели сохранен в best_model.txt")

print("\nОбучение завершено успешно!")