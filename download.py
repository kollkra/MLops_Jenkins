import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# URL датасета California Housing
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"

print("Загрузка данных...")
df = pd.read_csv(url)

print(f"Исходный размер данных: {df.shape}")
print(df.head())

# Обработка пропущенных значений
print("\nОбработка пропущенных значений...")
df = df.dropna()

# Удаление выбросов (например, очень высокие значения median_house_value)
print("Удаление выбросов...")
df = df[df['median_house_value'] < 500001]

# Создание дополнительных признаков
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']

# Кодирование категориального признака ocean_proximity
print("Кодирование категориальных признаков...")
if 'ocean_proximity' in df.columns:
    label_encoder = LabelEncoder()
    df['ocean_proximity'] = label_encoder.fit_transform(df['ocean_proximity'])

# Сохранение очищенных данных
output_file = 'df_clear.csv'
df.to_csv(output_file, index=False)
print(f"\nДанные сохранены в {output_file}")
print(f"Финальный размер данных: {df.shape}")
print("\nСтатистика данных:")
print(df.describe())