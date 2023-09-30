import warnings

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning, )

data = pd.read_csv('turboaz_27_09_2023.csv')


def convert_column_to_datetime(data, column_name, new_column_name):
    data[new_column_name] = pd.to_datetime(data[column_name].str.replace('Yeniləndi: ', '', regex=True),
                                           format='%d.%m.%Y')
    return data


def clean_and_convert_view(data, column_name, new_column_name):
    data[new_column_name] = data[column_name].str.replace('Baxışların sayı: ', '', regex=True)
    data[new_column_name] = pd.to_numeric(data[new_column_name].str.replace('[^\d.]', '', regex=True), errors='coerce')
    return data


def map_values(data, column_name, mapping):
    data[column_name] = data[column_name].apply(lambda value: mapping.get(value, value))
    return data


def extract_numeric_value(data, column_name, new_column_name):
    data[new_column_name] = data[column_name].str.extract(r'(\d+\.\d+)').astype(float)
    return data


def separate_price_and_currency(data):
    def helper(price_str):
        price_str = price_str.replace(',', '')
        currency_codes = ['USD', 'EUR', 'GBP', 'AZN', 'JPY', 'RUB']
        for code in currency_codes:
            if code in price_str:
                currency = code
                price = price_str.replace(code, '').strip()
                return price, currency
        return price_str, 'AZN'

    data[['car_price', 'currency']] = data['price'].apply(helper).apply(pd.Series)
    data['car_price'] = pd.to_numeric(data['car_price'].str.replace('[^\d.]', '', regex=True), errors='coerce')
    data['car_price'] = data.apply(lambda row: row['car_price'] * 1.7 if row['currency'] == 'USD' else (
        row['car_price'] * 1.8 if row['currency'] == 'EUR' else row['car_price']), axis=1)
    return data


data = convert_column_to_datetime(data, 'update', 'date')
data = clean_and_convert_view(data, 'views', 'views_cleaned')
data = map_values(data, 'ban_type', {
    'Offroader / SUV': 'SUV',
    'Sedan': 'Sedan',
    'Hetçbek': 'Hatchback',
    'Universal': 'Station Wagon',
    'Liftbek': 'Liftback',
    'Yük maşını': 'Truck',
    'Furqon': 'Van',
    'Minivan': 'Minivan',
    'Kupe': 'Coupe',
    'Motosiklet': 'Motorcycle',
    'Pikap': 'Pickup',
    'Dartqı': 'Convertible',
    'Mikroavtobus': 'Microbus',
    'Moped': 'Moped',
    'Avtobus': 'Bus',
    'Kabriolet': 'Convertible',
    'Van': 'Van',
    'Rodster': 'Roadster',
    'Kvadrosikl': 'Quad Bike'
})
data = map_values(data, 'colour', {
    'Ağ': 'Silver',
    'Qara': 'Black',
    'Gümüşü': 'Silver',
    'Göy': 'Blue',
    'Yaş Asfalt': 'Gray',
    'Boz': 'Brown',
    'Tünd qırmızı': 'Dark_Red',
    'Qırmızı': 'Red',
    'Yaşıl': 'Green',
    'Mavi': 'Blue',
    'Bej': 'Beige',
    'Qızılı': 'Gold',
    'Qəhvəyi': 'Brown',
    'Narıncı': 'Orange',
    'Sarı': 'Yellow',
    'Bənövşəyi': 'Purple',
    'Çəhrayı': 'Pink'
})
data = extract_numeric_value(data, 'engine', 'engine_power')
data = clean_and_convert_view(data, 'ride', 'ride_km')
data = map_values(data, 'transmission', {
    'Avtomat': 'Automatic',
    'Mexaniki': 'Manual',
    'Variator': 'CVT',
    'Robotlaşdırılmış': 'Automated_Manual'
})
data = map_values(data, 'gear', {
    'Tam': 'Full',
    'Ön': 'Front',
    'Arxa': 'Rear'
})
data = map_values(data, 'is_new', {
    'Bəli': 'Yes',
    'Xeyr': 'No'
})
data = separate_price_and_currency(data)

original_features = ['city', 'make', 'model', 'year', 'ban_type', 'colour', 'engine_power', 'ride_km', 'transmission',
                     'gear', 'is_new']
target = 'car_price'
data_copy = pd.get_dummies(data[original_features + [target]])
X = data_copy.drop(columns=[target])
y = data_copy[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)
model = XGBRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
print("Model perforamnce metrics")
print("-----------------------")
print(f"R-squared: {r2:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print("-----------------------")
joblib.dump(model, 'xgboost_model.pkl')
loaded_model = joblib.load('xgboost_model.pkl')
new_data_point = {
    'city': 'Bakı',
    'make': 'Jeep',
    'model': 'Wrangler',
    'year': 2022,
    'ban_type': 'SUV',
    'colour': 'Brown',
    'engine_power': 2.0,
    'ride_km': 0,
    'transmission': 'Automatic',
    'gear': 'Full',
    'is_new': 'Yes'}
new_data_df = pd.DataFrame([new_data_point])
categorical_columns = ['city', 'make', 'model', 'ban_type', 'colour', 'transmission', 'gear', 'is_new']
new_data_df = pd.get_dummies(new_data_df, columns=categorical_columns)
missing_columns = set(data_copy) - set(new_data_df.columns)
missing_columns_list = list(missing_columns)
zeros_df = pd.DataFrame(0, index=new_data_df.index, columns=missing_columns_list)
new_data_df = pd.concat([new_data_df, zeros_df], axis=1)
new_data_df = new_data_df[data_copy.columns].drop(columns=['car_price'])
predictions = loaded_model.predict(new_data_df)
print("-----------------------")
print("Inputs :")
print(new_data_point)
print("                       ")
print("Real Car Price      :", round(data['car_price'][0]))
print("Predicted Car Price :", round(predictions[0]))
print("Difference          :", round(data['car_price'][0] - predictions[0]))
print("-----------------------")
