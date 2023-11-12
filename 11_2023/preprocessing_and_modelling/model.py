import warnings
from preprocessor import read_and_preprocess_data, train_and_save_model
warnings.filterwarnings("ignore", category=FutureWarning, )

if __name__ == "__main__":
    try:
        print("Reading and preprocessing data...")
        data_file_paths = ['turboaz_27_09_2023.csv', '08112023.csv']
        original_features = ['city', 'make', 'model', 'year', 'ban_type', 'colour', 'engine_power', 'ride_km', 'transmission', 'gear', 'is_new']
        target = 'car_price'
        data = read_and_preprocess_data(data_file_paths)
        print("Training and saving the model...")
        trained_model, predictions = train_and_save_model(data, original_features, target)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
