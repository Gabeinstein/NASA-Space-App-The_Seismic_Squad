import os
import pandas as pd

def dataReader(i, folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')] 
    base_file_name = csv_files[i] 
    full_file_name = os.path.join(folder_path, base_file_name)  
    data = pd.read_csv(full_file_name) 
    return data, full_file_name
 