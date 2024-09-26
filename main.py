import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


@st.cache_data
def load_data():
    return pd.read_csv('UNSW_NB15_training-set.csv')

def preprocess_data(df):
    df.dropna(inplace=True)
    df.drop(['id'], axis=1, inplace=True)
    df['state'] = df['state'].replace('-','other')
    df['service'] = df['service'].replace('-','other')
    features_to_drop = ['ct_state_ttl', 'ct_dst_sport_ltm', 'ct_src_ltm']
    df = df.drop(features_to_drop, axis=1)
    df = pd.get_dummies(df, columns=['proto', 'service'], drop_first=True)
    label_encoder = LabelEncoder()
    df['state'] = label_encoder.fit_transform(df['state'])
    df['attack_cat'] = label_encoder.fit_transform(df['attack_cat'])
    df['load_interaction'] = df['sload'] * df['dload']
    df['total_bytes'] = df['sbytes'] + df['dbytes']
    df['pkt_flow_ratio'] = df['spkts'] / (df['dpkts'] + 1)
    df['bytes_diff'] = df['sbytes'] - df['dbytes']
    df['bytes_ratio'] = df['sbytes'] / (df['dbytes'] + 1)
    df['ttl_diff'] = df['sttl'] - df['dttl']
    df['jitter_diff'] = df['sjit'] - df['djit']
    df['jitter_ratio'] = df['sjit'] / (df['djit'] + 1)
    df['tcp_time_diff'] = df['synack'] - df['ackdat']
    return df.drop('label', axis=1), df['label']

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = XGBClassifier(learning_rate=0.01, max_depth=3, n_estimators=50)
    model.fit(X_train_scaled, y_train)
    return model, scaler

def main():
    st.title('Network Traffic Anomaly Detection')
    data = load_data()
    X, y = preprocess_data(data)
    model, scaler = train_model(X, y)

    # Define input fields based on the dataset
    input_data = {}

    # Add input fields
    input_data['srcip'] = st.text_input('Source IP Address')
    input_data['sport'] = st.number_input('Source Port Number', min_value=0)
    input_data['dstip'] = st.text_input('Destination IP Address')
    input_data['dsport'] = st.number_input('Destination Port Number', min_value=0)
    input_data['proto'] = st.selectbox('Protocol', ['TCP', 'UDP', 'ICMP', 'other'])
    input_data['state'] = st.selectbox('State', ['FIN', 'CON', 'REQ', 'RSTO', 'other'])
    input_data['dur'] = st.number_input('Duration', min_value=0.0)
    input_data['spkts'] = st.number_input('Source Packets', min_value=0)
    input_data['dpkts'] = st.number_input('Destination Packets', min_value=0)
    input_data['sbytes'] = st.number_input('Source Bytes', min_value=0)
    input_data['dbytes'] = st.number_input('Destination Bytes', min_value=0)
    input_data['sttl'] = st.number_input('Source Time to Live (TTL)', min_value=0)
    input_data['dttl'] = st.number_input('Destination Time to Live (TTL)', min_value=0)
    input_data['sload'] = st.number_input('Source Load', min_value=0.0)
    input_data['dload'] = st.number_input('Destination Load', min_value=0.0)
    input_data['sloss'] = st.number_input('Source Loss', min_value=0)
    input_data['dloss'] = st.number_input('Destination Loss', min_value=0)
    input_data['service'] = st.selectbox('Service', ['http', 'dns', 'smtp', 'ftp', 'other'])
    input_data['sjit'] = st.number_input('Source Jitter', min_value=0.0)
    input_data['djit'] = st.number_input('Destination Jitter', min_value=0.0)
    input_data['synack'] = st.number_input('SYN-ACK', min_value=0.0)
    input_data['ackdat'] = st.number_input('ACK-DATA', min_value=0.0)

    # Load the trained model and scaler
    model, scaler = load_model_and_scaler()

    # Define the columns that your model expects
    model_columns = ['sport', 'dsport', 'proto', 'state', 'dur', 'spkts', 'dpkts',
                     'sbytes', 'dbytes', 'sttl', 'dttl', 'sload', 'dload', 'sloss',
                     'dloss', 'service', 'sjit', 'djit', 'synack', 'ackdat']

    # When the user clicks the Predict button
    if st.button('Predict'):
        st.write("Processing the following input data:")
        st.write(input_data)
        
        # Prepare the input data
        input_df = prepare_input_data(input_data, model_columns)
        
        # Scale the input data
        input_scaled = scaler.transform(input_df)
        
        # Make a prediction
        prediction = model.predict(input_scaled)
        prediction_label = 'Attack' if prediction == 1 else 'Normal'
        
        # Display the result
        st.subheader(f'Prediction: {prediction_label}')

if __name__ == "__main__":
    main()