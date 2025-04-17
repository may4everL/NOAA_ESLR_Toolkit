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
    'A-7-6':9250
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
    'A-7-6':6261
}

def generate_Mr(gwt_vals, soil_type='A-2-4'):
    a, b, c = 2.12, 93.18, 20177.52
    gwt_vals_cm = np.array(gwt_vals) * 2.54
    if soil_type == 'A-1-a':
        a, b, c = -2.2124, 481.23, 25107.98
    if soil_type == 'A-1-b':
        a, b, c = -1.1682, 400.66, 23910.03
    if soil_type == 'A-2-4':
        a, b, c = 2.12, 93.18, 20177.52
    if soil_type == 'A-2-5':
        a, b, c = -1.1045, 302.22, 18070.04
    if soil_type == 'A-2-6':
        a, b, c = -0.973, 274.29, 16640.73
    if soil_type == 'A-2-7':
        a, b, c = -0.9794, 260.56, 15695.62
    if soil_type == 'A-3':
        a, b, c = 1.9, 79.53, 18790.98
    if soil_type == 'A-4':
        a, b, c = 0.0424, 12.17, 15533.9
    if soil_type == 'A-5':
        a, b, c = 0.0823, 7.97, 12915.11
    if soil_type == 'A-6':
        a, b, c = 0.0088, 6.97, 12002.95
    if soil_type == 'A-7-5':
        a, b, c = 0.00146, 3.295, 8283.2
    if soil_type == 'A-7-6':
        a, b, c = 0.00021, 1.65, 6200.98
    Mrs = [a*(gwt)**2 + b*(gwt) + c for gwt in gwt_vals_cm]
    # print(f'before cutoff: {Mrs}')
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