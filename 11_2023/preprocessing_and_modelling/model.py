import warnings
from preprocessor import preprocess_data, train_and_save_model,original_features,target
warnings.filterwarnings("ignore", category=FutureWarning, )

if __name__ == "__main__":
    try:
        print("Reading and preprocessing data...")
        data_file_paths = ['turboaz_27_09_2023.csv', '08112023.csv']
        data = preprocess_data(data_file_paths)
        data[original_features].to_excel('preprocess_data.xlsx',index=False)
        print("Training and saving the model...")
        trained_model, predictions = train_and_save_model(data, original_features, target)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
