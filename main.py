import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    st.title('Anomaly Detection Predictor')
    data = load_data()
    X, y = preprocess_data(data)
    model, scaler = train_model(X, y)

    st.header('Enter Your Information')
    input_data = {}
    input_data['company_name'] = st.text_input('Company Name')
    input_data['foundation_date'] = st.date_input('Foundation Date')
    input_data['age_first_funding_year'] = st.number_input('Age at First Funding (years)', min_value=0.0)
    input_data['age_last_funding_year'] = st.number_input('Age at Last Funding (years)', min_value=0.0)
    input_data['relationships'] = st.number_input('Number of Relationships', min_value=0)
    input_data['funding_rounds'] = st.number_input('Number of Funding Rounds', min_value=0)
    input_data['funding_total_usd'] = st.number_input('Total Funding (USD)', min_value=0)
    input_data['milestones'] = st.number_input('Number of Milestones', min_value=0)
    input_data['has_VC'] = st.checkbox('Has Venture Capital Funding')
    input_data['has_angel'] = st.checkbox('Has Angel Funding')
    input_data['has_roundA'] = st.checkbox('Has Round A Funding')
    input_data['has_roundB'] = st.checkbox('Has Round B Funding')
    input_data['has_roundC'] = st.checkbox('Has Round C Funding')
    input_data['has_roundD'] = st.checkbox('Has Round D Funding')
    input_data['avg_participants'] = st.number_input('Average Number of Funding Participants', min_value=0.0)
    input_data['is_top500'] = st.checkbox('Is in Top 500')

    if st.button('Predict Success'):
        prediction_input = prepare_input_data(input_data, X.columns)
        prediction_input_scaled = scaler.transform(prediction_input)
        probability = model.predict_proba(prediction_input_scaled)[0][1]
        st.subheader('Prediction Result')
        st.success(f'The Probability of success: {probability:.2%}')

def prepare_input_data(input_data, columns):
    df = pd.DataFrame(input_data, index=[0])
    df['has_RoundABCD'] = ((df[['has_roundA', 'has_roundB', 'has_roundC', 'has_roundD']] == 1).any(axis=1)).astype(int)
    df['has_Investor'] = ((df[['has_VC', 'has_angel']] == 1).any(axis=1)).astype(int)
    df['has_Seed'] = ((df['has_RoundABCD'] == 0) & (df['has_Investor'] == 1)).astype(int)
    df['invalid_startup'] = ((df['has_RoundABCD'] == 0) & (df['has_VC'] == 0) & (df['has_angel'] == 0)).astype(int)
    
    df = pd.get_dummies(df, columns=['state', 'category', 'city'], prefix=['State', 'category', 'City'])
    
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    
    return df[columns]

if __name__ == "__main__":
    main()