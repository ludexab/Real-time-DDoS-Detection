import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("datasets/cicddos2019_dataset.csv")

selected_features = [
    "Flow Duration", "Fwd Packets Length Total", "Bwd Packets Length Total", "Fwd Packet Length Max", 
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std", "Flow Bytes/s", 
    "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", 
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd Header Length", 
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Packet Length Min", "Packet Length Max", 
    "Packet Length Mean", "Packet Length Std", "Avg Packet Size", "Avg Fwd Segment Size", 
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Init Fwd Win Bytes", "Init Bwd Win Bytes", 
    "Fwd Act Data Packets", "Fwd Seg Size Min", "Class"
]

data = data[selected_features]

# Split into Offline Training (60%) and Temp (40%)
offline_train, temp = train_test_split(data, test_size=0.4, random_state=42)

# Split Temp into Online Training (75% of temp) and Validation Holdout (25% of temp)
online_train, validation_holdout = train_test_split(temp, test_size=0.25, random_state=42)

# Check sizes
print(f"Offline Training size: {len(offline_train)}")
print(f"Online Training size: {len(online_train)}")
print(f"Validation Holdout size: {len(validation_holdout)}")

# Save the offline training data
offline_train.to_csv("datasets/cicddos2019_offline_train.csv", index=False)

# Save the online training data
online_train.to_csv("datasets/cicddos2019_online_train.csv", index=False)

# Save the validation holdout data
validation_holdout.to_csv("datasets/cicddos2019_validation_holdout.csv", index=False)
