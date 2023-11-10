import warnings
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
warnings.filterwarnings("ignore", category=FutureWarning, )

data_1 = pd.read_csv('turboaz_27_09_2023.csv')
data_2 = pd.read_csv('08112023.csv')
data = pd.concat([data_1, data_2]).drop_duplicates().dropna()
data.columns = data.columns.str.replace('[^a-zA-Z0-9_]', '_', regex=True)
def convert_column_to_datetime(data, column_name, new_column_name):
    data[new_column_name] = pd.to_datetime(data[column_name].str.replace('Yeniləndi: ','', regex=True),format='%d.%m.%Y')
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
def translate_to_english(city_name):
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
    return city_mapping.get(city_name, city_name)

def remove_underscores_from_columns(df):
    for column in df.columns:
        df.rename(columns={column: column.replace('_', '')}, inplace=True)



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
data['city'] = data['city'].apply(translate_to_english)

original_features = ['city', 'make', 'model', 'year', 'ban_type', 'colour', 'engine_power', 'ride_km', 'transmission','gear', 'is_new']
target = 'car_price'
data_copy = pd.get_dummies(data[original_features + [target]])
data_copy.to_excel('datacleaned.xlsx')
X = data_copy.drop(columns=['car_price'])
y = data_copy['car_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)
model = XGBRegressor(learning_rate = 0.4 ,max_depth=4 ,n_estimators = 1200)
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