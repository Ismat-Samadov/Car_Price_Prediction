import warnings

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
warnings.filterwarnings("ignore", category=FutureWarning, )

def extract_engine_power(data, column_name, new_column_name):
    data[new_column_name] = data[column_name].str.extract(r'(\d+\.\d+)').astype(float)
    return data

def extract_ride_km(data, column_name, new_column_name):
    data[new_column_name] = data[column_name].str.replace('[^\d.]', '', regex=True).astype(float)
    return data

def map_values(data, column_name, mapping):
    data[column_name] = data[column_name].apply(lambda value: mapping.get(value, value))
    return data

def map_city(data, column_name):
    city_mapping = {
        'Bakı': 'Baku',
        'Gəncə': 'Ganja',
        'Sumqayıt': 'Sumgait',
        'Lənkəran': 'Lankaran',
        'Şamaxı': 'Shamakhi',
        'Sabirabad': 'Sabirabad',
        'Salyan': 'Salyan',
        'Masallı': 'Masalli',
        'Şirvan': 'Shirvan',
        'İsmayıllı': 'Ismayilli',
        'Biləsuvar': 'Bilasuvar',
        'Ağdaş': 'Agdash',
        'Tovuz': 'Tovuz',
        'Goranboy': 'Goranboy',
        'Şəki': 'Sheki',
        'Xırdalan': 'Khirdalan',
        'Ağcabədi': 'Agjabadi',
        'Quba': 'Quba',
        'Balakən': 'Balakan',
        'Şəmkir': 'Shamkir',
        'Qazax': 'Qazakh',
        'Mingəçevir': 'Mingachevir',
        'Bərdə': 'Barda',
        'Saatlı': 'Saatli',
        'Xaçmaz': 'Khachmaz',
        'Kürdəmir': 'Kurdamir',
        'Göyçay': 'Goychay',
        'Neftçala': 'Neftchala',
        'Ağsu': 'Agsu',
        'Qəbələ': 'Qabala',
        'Tərtər': 'Tartar',
        'Naxçıvan': 'Nakhchivan',
        'Astara': 'Astara',
        'Qax': 'Qakh',
        'Siyəzən': 'Siyezen',
        'Lerik': 'Lerik',
        'Yevlax': 'Yevlakh',
        'Gədəbəy': 'Gedabay',
        'İmişli': 'Imishli',
        'Zaqatala': 'Zaqatala',
        'Göygöl': 'Goygol',
        'Yardımlı': 'Yardimli',
        'Şabran': 'Shabran',
        'Qusar': 'Qusar',
        'Beyləqan': 'Beylagan',
        'Samux': 'Samukh',
        'Ağdam': 'Agdam',
        'Cəlilabad': 'Jalilabad',
        'Ağstafa': 'Agstafa',
        'Hacıqabul': 'Hajigabul',
        'Ucar': 'Ujar',
        'Füzuli': 'Fuzuli',
        'Qobustan': 'Gobustan',
        'Xudat': 'Khudat',
        'Oğuz': 'Oguz',
        'Zərdab': 'Zardab',
        'Babək': 'Babek',
        'Göytəpə': 'Goytepe',
        'Cəbrayıl': 'Jebrail',
        'Daşkəsən': 'Dashkasan',
        'Naftalan': 'Naftalan',
        'Dəliməmmədli': 'Delimammadli',
        'Şərur': 'Sharur',
        'Horadiz': 'Horadiz',
        'Ordubad': 'Ordubad',
        'Xızı': 'Khizi',
    }
    data[column_name] = data[column_name].apply(lambda value: city_mapping.get(value, value))
    return data

def map_ban_type(data, column_name):
    ban_type_mapping = {
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
    }

    data[column_name] = data[column_name].apply(lambda value: ban_type_mapping.get(value, value))
    return data

def map_colour(data, column_name):
    colour_mapping = {
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
    }
    data[column_name] = data[column_name].apply(lambda value: colour_mapping.get(value, value))
    return data

def map_transmission(data, column_name):
    transmission_mapping = {
        'Avtomat': 'Automatic',
        'Mexaniki': 'Manual',
        'Variator': 'CVT',
        'Robotlaşdırılmış': 'Automated_Manual'
    }
    data[column_name] = data[column_name].apply(lambda value: transmission_mapping.get(value, value))
    return data

def map_gear(data, column_name):
    gear_mapping = {
        'Tam': 'Full',
        'Ön': 'Front',
        'Arxa': 'Rear'
    }
    data[column_name] = data[column_name].apply(lambda value: gear_mapping.get(value, value))
    return data

def map_is_new(data, column_name):
    is_new_mapping = {
        'Bəli': 'Yes',
        'Xeyr': 'No'
    }
    data[column_name] = data[column_name].apply(lambda value: is_new_mapping.get(value, value))
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

def preprocess_data(data):
    data = map_city(data, 'city')
    data = map_ban_type(data, 'ban_type')
    data = map_colour(data, 'colour')
    data = map_transmission(data, 'transmission')
    data = map_gear(data, 'gear')
    data = map_is_new(data, 'is_new')
    data = extract_engine_power(data, 'engine', 'engine_power')
    data = extract_ride_km(data, 'ride', 'ride_km')
    data = separate_price_and_currency(data)
    return data


def split_data(data, original_features, target, test_size=0.2, random_state=47):
    data = data.dropna()
    categorical_columns = data[original_features].select_dtypes(include=['object']).columns
    data.loc[:, categorical_columns] = data[categorical_columns].astype('category')
    data_copy = pd.get_dummies(data[original_features + [target]], drop_first=True, sparse=True)
    X = data_copy.drop(columns=[target])
    y = data_copy[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_xgboost_model(X_train, y_train, learning_rate=0.4, max_depth=4, n_estimators=1200):
    model = XGBRegressor(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    print("Model performance metrics")
    print("-----------------------")
    print(f"R-squared: {r2:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print("-----------------------")
    return y_pred


def read_and_preprocess_data(file_paths):
    data = pd.concat([pd.read_csv(file_path) for file_path in file_paths]).drop_duplicates().dropna()
    data = preprocess_data(data)
    return data


def train_and_save_model(data, original_features, target, model_file_path='xgboost_model.pkl'):
    X_train, X_test, y_train, y_test = split_data(data, original_features, target)
    model = train_xgboost_model(X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)
    joblib.dump(model, model_file_path)

    return model, y_pred


original_features = ['city', 'make', 'model', 'year', 'ban_type', 'colour', 'engine_power', 'ride_km', 'transmission',
                     'gear', 'is_new']
