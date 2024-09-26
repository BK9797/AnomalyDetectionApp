import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


@st.cache_data
def load_data():
    return pd.read_csv('UNSW_NB15_training-set.csv')

def preprocess_data(df):
    df.dropna(inplace=True)
    df.drop(['id'], axis=1, inplace=True)
    df['state'] = df['state'].replace('-', 'other')
    df['service'] = df['service'].replace('-', 'other')
    features_to_drop = ['ct_state_ttl', 'ct_dst_sport_ltm', 'ct_src_ltm']
    df = df.drop(features_to_drop, axis=1)
    df = pd.get_dummies(df, columns=['proto', 'service'], drop_first=True)
    label_encoder = LabelEncoder()
    df['state'] = label_encoder.fit_transform(df['state'])
    df['attack_cat'] = label_encoder.fit_transform(df['attack_cat'])
    return df.drop('label', axis=1), df['label']

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = XGBClassifier(learning_rate=0.01, max_depth=3, n_estimators=50)
    model.fit(X_train_scaled, y_train)
    return model, scaler

def load_model_and_scaler():
    data = load_data()
    X, y = preprocess_data(data)
    return train_model(X, y)

def prepare_input_data(input_data, model_columns):
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df, columns=['proto', 'state', 'service'], drop_first=True)
    
    # Ensure all model columns are present in the input
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Add missing columns with default value 0

    # Reorder columns to match model's expected input format
    input_df = input_df[model_columns]
    
    return input_df

def main():
    st.title('Network Traffic Anomaly Detection')

    # Load the trained model and scaler
    model, scaler = load_model_and_scaler()

    # Define input fields for user
    input_data = {
        'proto': st.selectbox('Protocol', ['TCP', 'UDP', 'ICMP', 'other']),
        'state': st.selectbox('State', ['FIN', 'CON', 'REQ', 'RSTO', 'other']),
        'dur': st.number_input('Duration', min_value=0.0),
        'spkts': st.number_input('Source Packets', min_value=0),
        'dpkts': st.number_input('Destination Packets', min_value=0),
        'sbytes': st.number_input('Source Bytes', min_value=0),
        'dbytes': st.number_input('Destination Bytes', min_value=0),
        'sttl': st.number_input('Source TTL', min_value=0),
        'dttl': st.number_input('Destination TTL', min_value=0),
        'sload': st.number_input('Source Load', min_value=0.0),
        'dload': st.number_input('Destination Load', min_value=0.0),
        'sloss': st.number_input('Source Loss', min_value=0),
        'dloss': st.number_input('Destination Loss', min_value=0),
        'service': st.selectbox('Service', ['http', 'dns', 'smtp', 'ftp', 'other']),
        'sjit': st.number_input('Source Jitter', min_value=0.0),
        'djit': st.number_input('Destination Jitter', min_value=0.0),
        'synack': st.number_input('SYN-ACK', min_value=0.0),
        'ackdat': st.number_input('ACK-DATA', min_value=0.0)
    }

    # Define the columns your model expects
    model_columns = ['proto', 'state', 'dur', 'spkts', 'dpkts', 
                     'sbytes', 'dbytes', 'sttl', 'dttl', 'sload', 'dload', 
                     'sloss', 'dloss', 'service', 'sjit', 'djit', 'synack', 'ackdat']

    # # Prediction button
    # if st.button('Predict'):
    #     st.write("Processing the following input data:")
    #     st.write(input_data)

        # # Prepare and scale the input data
        # input_df = prepare_input_data(input_data, model_columns)
        # input_scaled = scaler.transform(input_df)
        
        # Make a prediction
    prediction = model.predict(input_data)
    prediction_label = 'Attack' if prediction == 1 else 'Normal'
        
        # Display the result
    st.subheader(f'Prediction: {prediction_label}')

if __name__ == "__main__":
    main()
