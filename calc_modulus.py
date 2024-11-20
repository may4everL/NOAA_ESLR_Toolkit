import pandas as pd
import numpy as np
from scipy.spatial import distance
import pickle

Mr_opts = {
    'A-1-a':46000,
    'A-1-b':40000,
    'A-2-4':36000,
    'A-2-5':32000,
    'A-2-6':28000,
    'A-2-7':24000,
    'A-3':18000,
    'A-4':12000,
    'A-5':8000,
    'A-6':6000,
    'A-7-5':4500,
    'A-7-6':3500
}
Mr_sats = {
    'A-1-a':36800,
    'A-1-b':32000,
    'A-2-4':27000,
    'A-2-5':24000,
    'A-2-6':21000,
    'A-2-7':18000,
    'A-3':14000,
    'A-4':9000,
    'A-5':5000,
    'A-6':4000,
    'A-7-5':2000,
    'A-7-6':1200
}

def generate_Mr(gwt_vals, soil_type='A-2-4'):
    a, b, c = 0.000133, -0.0123, 0.928
    Mr_initial = Mr_opts[soil_type]
    if soil_type == 'A-1-a':
        a, b, c = -1e-4, 1.95e-2, 5.592e-1
    if soil_type == 'A-1-b':
        a, b, c = -1e-4, 2.01e-2, 5.598e-1
    if soil_type == 'A-2-4':
        a, b, c = -1e-4, 2.16e-2, 5.601e-1
    if soil_type == 'A-2-5':
        a, b, c = -1e-4, 2.06e-2, 5.610e-1
    if soil_type == 'A-2-6':
        a, b, c = -7.33e-5, 1.81e-2, 5.470e-1
    if soil_type == 'A-2-7':
        a, b, c = 2.06e-5, 9.71e-3, 5.540e-1
    if soil_type == 'A-3':
        a, b, c = -1.84e-7, 7.03e-4, 6.140e-1
    if soil_type == 'A-4':
        a, b, c = -8.05e-5, 2.23e-2, 4.11e-1
    if soil_type == 'A-5':
        a, b, c = -7.00e-5, 2.17e-2, 4.214e-1
    if soil_type == 'A-6':
        a, b, c = 4.00e-5, 9.40e-3, 3.614e-1
    if soil_type == 'A-7-5':
        a, b, c = 7.36e-6, 7.57e-3, 2.550e-1
    if soil_type == 'A-7-6':
        a, b, c = -2.88e-6, 5.95e-3, 2.110e-1
    Mrs = [Mr_initial * (a*(gwt)**2 + b*(gwt) + c) for gwt in gwt_vals]
    print(f'before cutoff: {Mrs}')
    Mrs = [min(max(x, Mr_sats[soil_type]), Mr_opts[soil_type]) for x in Mrs] # apply cutoff from Mr_sat to Mr_opt
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
        mr = knn_predict_with_weights(df, input_params, k=k, weights=weights)
        mr_values.append(mr)
    return mr_values