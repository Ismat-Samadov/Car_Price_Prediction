import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

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
    return data


def train_xgboost_model(data, original_features, target, test_size=0.2, random_state=42):
    data_copy = data[original_features + target].copy()
    data_copy = pd.get_dummies(data_copy,
                               columns=['city', 'make', 'model', 'ban_type', 'colour', 'transmission', 'gear',
                                        'is_new'])

    features = data_copy.columns.tolist()
    features.remove(target[0])

    X = data_copy[features]
    y = data_copy[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = XGBRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    metrics = {
        'r2_score': r2,
        'mean_squared_error': rmse,
        'mean_absolute_error': mae
    }

    return metrics


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
    'Variator': 'CVT (Continuously_Variable_Transmission)',
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

main_columns = ['views_cleaned', 'city', 'make', 'model', 'year', 'ban_type', 'colour', 'engine_power', 'ride_km',
                'transmission', 'gear', 'is_new', 'car_price']

data = pd.DataFrame(data[main_columns])

original_features = ['views_cleaned', 'city', 'make', 'model', 'year', 'ban_type', 'colour', 'engine_power', 'ride_km',
                     'transmission', 'gear', 'is_new']
target = ['car_price']

metrics = train_xgboost_model(data, original_features, target)
print(metrics)
