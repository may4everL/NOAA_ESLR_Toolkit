import pandas as pd
import numpy as np
from scipy.spatial import distance
import pickle

Mr_opts = {
    'A-1-a':40250,
    'A-1-b':37750,
    'A-2-4':32750,
    'A-2-5':28500,
    'A-2-6':26250,
    'A-2-7':24750,
    'A-3':30500,
    'A-4':25250,
    'A-5':21250,
    'A-6':18750,
    'A-7-5':12750,
    'A-7-6':9250,
    1:40250,
    1:37750,
    2:32750,
    2:28500,
    2:26250,
    2:24750,
    3:30500,
    4:25250,
    5:21250,
    5:18750,
    6:12750,
    6:9250
}
Mr_sats = {
    'A-1-a':25582,
    'A-1-b':23993,
    'A-2-4':20815,
    'A-2-5':18114,
    'A-2-6':16684,
    'A-2-7':15730,
    'A-3':19385,
    'A-4':15544,
    'A-5':13026,
    'A-6':12155,
    'A-7-5':8477,
    'A-7-6':6261,
    1:25582,
    1:23993,
    2:20815,
    2:18114,
    2:16684,
    2:15730,
    3:19385,
    4:15544,
    5:13026,
    5:12155,
    6:8477,
    6:6261
}

def generate_Mr(gwt_vals, soil_type='A-2-4'):
    a, b, c = 2.12, 93.18, 20177.52
    # gwt_vals_cm = np.array(gwt_vals) * 2.54
    if soil_type == 'A-1-a' or 1:
        a, b, c = -2.36e-4, 2.80e-2, 0.63
    if soil_type == 'A-1-b' or 1:
        a, b, c = -2.00e-4, 2.70e-2, 0.63
    if soil_type == 'A-2-4' or 2:
        a, b, c =  4.17e-4, 7.20e-3, 0.62
    if soil_type == 'A-2-5' or 2:
        a, b, c = -2.50e-4, 2.69e-2, 0.63
    if soil_type == 'A-2-6' or 2:
        a, b, c = -2.39e-4, 2.65e-2, 0.63
    if soil_type == 'A-2-7' or 2:
        a, b, c = -2.55e-4, 2.67e-2, 0.63
    if soil_type == 'A-3' or 3:
        a, b, c =  3.90e-4, 6.90e-3, 0.62
    if soil_type == 'A-4' or 4:
        a, b, c =  1.11e-5, 1.20e-3, 0.61
    if soil_type == 'A-5' or 5:
        a, b, c =  1.55e-5, 1.20e-3, 0.60
    if soil_type == 'A-6' or 5:
        a, b, c =  3.10e-6, 9.00e-4, 0.64
    if soil_type == 'A-7-5' or 6:
        a, b, c =  7.00e-7, 7.00e-4, 0.66
    if soil_type == 'A-7-6' or 6:
        a, b, c =  1.00e-7, 5.00e-4, 0.67
    Mr_sat = Mr_sats[soil_type]
    Mr_opt = Mr_opts[soil_type]
    Mrs = [(a*(gwt)**2 + b*(gwt) + c) * Mr_opt for gwt in gwt_vals]
    # print(f'before cutoff: {Mrs}')
    Mrs = [min(max(x, Mr_sat), Mr_opt) for x in Mrs] # apply cutoff from Mr_sat to Mr_opt
    return Mrs


with open('Models/modulus_table.pkl', 'rb') as file:
    # Load the data from the file
    df = pickle.load(file)

def knn_predict_with_weights(df, input_params, k=5, weights=None):
    """
    Predict the Equivalent Modulus using KNN with custom feature weights.
    
    Parameters:
    - df: pandas DataFrame containing the dataset.
        Expected columns: ['Surface Thickness', 'Base Thickness', 'GWT', 'Base Type', 'Subgrade Type', 'Equivalent Modulus']
    - input_params: dict containing the input values for prediction.
        Example: {'Surface Thickness': 175, 'Base Thickness': 225, 'GWT': 7, 'Base Type': 1, 'Subgrade Type': 2}
    - k: Number of nearest neighbors.
    - weights: dict specifying weights for each feature.
        Example: {'Surface Thickness': 1.0, 'Base Thickness': 1.0, 'GWT': 1.0, 'Base Type': 2.0, 'Subgrade Type': 2.0}
    
    Returns:
    - Predicted Equivalent Modulus (float).
    """
    # Default weights if not provided
    if weights is None:
        weights = {'Surface Thickness': 1.0, 'Base Thickness': 1.0, 'GWT': 1.0, 'Base Type': 5.0, 'Subgrade Type': 5.0}
    
    # Separate features and target
    features = ['Surface Thickness', 'Base Thickness', 'GWT', 'Base Type', 'Subgrade Type']
    target = 'Equivalent Modulus'
    
    # Ensure all necessary columns are in the DataFrame
    missing_cols = set(features + [target]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in DataFrame: {missing_cols}")
    
    # Preprocess the data
    df_features = df[features].copy()
    df_target = df[target].copy()
    
    # Apply weights to features
    for col in features:
        if col in weights:
            df_features[col] = df_features[col] * weights[col]
            input_params[col] = input_params[col] * weights[col]
    
    # Convert input_params to a DataFrame
    input_vector = pd.DataFrame([input_params], columns=features)
    
    # Calculate distances
    distances = df_features.apply(lambda row: distance.euclidean(row, input_vector.iloc[0]), axis=1)
    
    # Get the indices of the k nearest neighbors
    nearest_neighbors = distances.nsmallest(k).index
    
    # Calculate the average of the target variable for the nearest neighbors
    predicted_modulus = df_target.iloc[nearest_neighbors].mean()
    
    return predicted_modulus

def generate_flooded_Mr(input_params_list, df=df, k=5, weights=None):
    """
    Predict the Equivalent Modulus (Mr) over the design years.

    Parameters:
    - df: pandas DataFrame containing the dataset.
    - input_params_list: list of input_params dictionaries for each year.
    - k: Number of nearest neighbors for KNN.
    - weights: dict specifying weights for each feature.

    Returns:
    - List of predicted Mr values, one for each year.
    """

    mr_values = []
    for input_params in input_params_list:
        cur_gwt = input_params['GWT']
        cur_soil_type = input_params['Subgrade Type']
        upper_limit = generate_Mr([cur_gwt], cur_soil_type)[0]
        # print(upper_limit)
        mr = knn_predict_with_weights(df, input_params, k=k, weights=weights)
        if mr > upper_limit:
            mr = upper_limit
        mr_values.append(mr)
    return mr_values