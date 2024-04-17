import pandas as pd
import os

def load_data(file_name):
    """
    load headlines data set from json file
    :param file_name: json file name
    """
    data_folder = '../data' # Data folder

    # Full path to the file
    file_path = os.path.join(data_folder, file_name)

    # Get the file extension
    file_extension = os.path.splitext(file_path)[1]

    # Load the file depending on its type
    if file_extension == '.json':
        df = pd.read_json(file_path, lines=True)
    elif file_extension == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .json or .csv.")

    # Check the loaded data
    print(df.head())
    return df # Return the loaded DataFrame

file_name = 'Sarcasm_Headlines_Dataset_v2.json' # as in the data folder

# Load and read data
df = load_data(file_name)
df.head()