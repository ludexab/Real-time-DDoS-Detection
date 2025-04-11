import pandas as pd
import numpy as np
import pickle
from river import stream, compose, preprocessing, metrics
from river.forest import ARFClassifier
# from kafka import KafkaProducer, KafkaConsumer
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File paths
offline_train_path = 'datasets/cicddos2019_offline_train.csv'
online_train_path = 'datasets/cicddos2019_online_train.csv'
holdout_validation_path = 'datasets/cicddos2019_validation_holdout.csv'
model_save_path = 'models/ARF-Model.pkl'

# Kafka configuration
kafka_topic = 'online_training'
kafka_broker = 'localhost:9092'

# Load datasets
def load_data(filepath):
    return pd.read_csv(filepath)

# Preprocessing function
def preprocess_data(df):
    scaler = preprocessing.StandardScaler()
    df.fillna(0, inplace=True)
    scaled_data = []

    for _, row in df.iterrows():
        feature_dict = row[:-1].to_dict()  # Extract features as a dictionary
        scaled_features = scaler.learn_one(feature_dict)  # Update scaler state
        scaled_row = scaler.transform_one(feature_dict)  # Scale the features
        scaled_row[df.columns[-1]] = row[-1]  # Add the label back
        scaled_data.append(scaled_row)

    return pd.DataFrame(scaled_data)

# Prepare the data for ARFClassifier
def prepare_stream(df):
    feature_columns = df.columns[:-1]  # All columns except the last one
    target_column = df.columns[-1]    # The last column is the target
    return stream.iter_pandas(df[feature_columns], y=df[target_column])

# Train the model offline
def train_offline(stream_data, model):
    logging.info("Starting offline training...")
    for x, y in stream_data:
        model.learn_one(x, y)
    logging.info("Offline training completed.")
    return model

# Train the model online
def train_online(model):
    logging.info("Starting online training...")
    consumer = KafkaConsumer(kafka_topic, bootstrap_servers=kafka_broker, auto_offset_reset='earliest')
    for message in consumer:
        record = json.loads(message.value)
        x, y = record['features'], record['Class']
        model.learn_one(x, y)
    logging.info("Online training completed.")
    return model

# Validate the model
def validate_model(stream_data, model):
    logging.info("Starting model validation...")
    metric = metrics.ClassificationReport()
    for x, y in stream_data:
        y_pred = model.predict_one(x)
        metric.update(y, y_pred)
    logging.info("Validation completed.")
    print(metric)

# Save the trained model
def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Model saved to {filepath}")

# Main function
def main():
    # Load and preprocess datasets
    offline_data = load_data(offline_train_path)
    online_data = load_data(online_train_path)
    holdout_data = load_data(holdout_validation_path)

    offline_data = preprocess_data(offline_data)
    holdout_data = preprocess_data(holdout_data)

    # Prepare streams
    offline_stream = prepare_stream(offline_data)
    holdout_stream = prepare_stream(holdout_data)

    # Initialize model
    model = ARFClassifier()

    # Offline training
    model = train_offline(offline_stream, model)

    # Save the model
    save_model(model, model_save_path)

    # # Simulate online training
    # producer = KafkaProducer(bootstrap_servers=kafka_broker, value_serializer=lambda v: json.dumps(v).encode('utf-8'))
    # for _, row in online_data.iterrows():
    #     record = {"features": row[:-1].to_dict(), "Class": row[-1]}
    #     producer.send(kafka_topic, value=record)
    # producer.close()

    # model = train_online(model)

    # Holdout validation
    validate_model(holdout_stream, model)

if __name__ == "__main__":
    main()
