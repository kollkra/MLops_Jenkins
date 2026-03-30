import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import os

def download_data():
    url = 'https://raw.githubusercontent.com/dayekb/Basic_ML_Alg/main/cars_moldova_no_dup.csv'
    df = pd.read_csv(url, delimiter=',')
    df.to_csv("cars.csv", index=False)
    return df

def clear_data(path2df):
    if not os.path.exists(path2df):
        print("Файл не найден!")
        return False
        
    df = pd.read_csv(path2df)
    
    cat_columns = ['Make', 'Model', 'Style', 'Fuel_type', 'Transmission']
    num_columns = ['Year', 'Distance', 'Engine_capacity(cm3)', 'Price(euro)']

    question_dist = df[(df.Year < 2021) & (df.Distance < 1100)]
    df = df.drop(question_dist.index)
    question_dist = df[(df.Distance > 1e6)]
    df = df.drop(question_dist.index)
    question_engine = df[df["Engine_capacity(cm3)"] < 200]
    df = df.drop(question_engine.index)
    question_engine = df[df["Engine_capacity(cm3)"] > 5000]
    df = df.drop(question_engine.index)
    question_price = df[(df["Price(euro)"] < 101)]
    df = df.drop(question_price.index)
    question_price = df[df["Price(euro)"] > 1e5]
    df = df.drop(question_price.index)
    question_year = df[df.Year < 1971]
    df = df.drop(question_year.index)

    df = df.reset_index(drop=True)  
    
    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_columns])
    Ordinal_encoded = ordinal.transform(df[cat_columns])
    df_ordinal = pd.DataFrame(Ordinal_encoded, columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]
    
    df.to_csv('df_clear.csv', index=False)
    return True

if __name__ == "__main__":
    download_data()
    clear_data("cars.csv")
    print("Data preparation finished.")
