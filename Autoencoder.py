import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# ==========================================
# 1. LOAD AND PREPROCESS DATA
# ==========================================
file_path = 'Pre_train_D_0.csv'

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found. Please place it in your project folder.")
    exit()

# Load the CSV
df = pd.read_csv(file_path)
print(f"File Loaded. Total Messages: {len(df)}")

# STEP A: Split the 'Data' string into 8 separate columns
# Your file has data like "06 25 05...". This turns it into 8 individual columns.
data_split = df['Data'].str.split(' ', expand=True)


# STEP B: Convert Hexadecimal (e.g., 'FF', '0A') to Integer (255, 10)
def hex_to_int(val):
    try:
        return int(str(val), 16)
    except:
        return 0  # Handle missing or non-hex values


print("Converting Hexadecimal data to Integers...")
numeric_features = data_split.map(hex_to_int).values

# STEP C: Scale the data (Mandatory for Autoencoders)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_features)

# STEP D: Convert to PyTorch Tensors
train_data = torch.tensor(scaled_data, dtype=torch.float32)
input_size = train_data.shape[1]  # This will be 8


# ==========================================
# 2. DEFINE THE AUTOENCODER ARCHITECTURE
# ==========================================
class CANMonitor(nn.Module):
    def __init__(self, input_dim):
        super(CANMonitor, self).__init__()
        # Encoder: 8 -> 16 -> 4 (Compression)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )
        # Decoder: 4 -> 16 -> 8 (Reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ==========================================
# 3. TRAINING SETUP
# ==========================================
model = CANMonitor(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
print("\nStarting Training on Normal Traffic...")
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(train_data)
    loss = criterion(outputs, train_data)

    # Backward pass
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")


# ==========================================
# 5. ANOMALY DETECTION FUNCTION
# ==========================================
def check_message(hex_string):
    """
    Input: A string like "FF 00 12 34 AA BB CC DD"
    Output: Health Status
    """
    model.eval()
    with torch.no_grad():
        # 1. Convert input string to numeric array
        raw_values = [int(x, 16) for x in hex_string.split(' ')]
        # 2. Scale using the training scaler
        scaled_input = scaler.transform([raw_values])
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32)

        # 3. Predict/Reconstruct
        reconstruction = model(input_tensor)
        error = torch.mean((input_tensor - reconstruction) ** 2).item()

        # 4. Determine Status (Threshold 0.8 is common for scaled data)
        threshold = 0.8
        if error > threshold:
            return f"🚨 ANOMALY DETECTED! (Score: {error:.4f})"
        else:
            return f"✅ Normal (Score: {error:.4f})"


# ==========================================
# 6. TEST THE SYSTEM
# ==========================================
print("\n--- System Monitoring Test ---")

# Example 1: A normal sample from your data (row 0)
normal_sample = "06 25 05 30 FF CF 71 55"
print(f"Testing Normal Sample: {check_message(normal_sample)}")

# Example 2: An anomaly (simulating values that were never seen in training)
fake_attack = "FF FF FF FF FF FF FF FF"
print(f"Testing Attack Sample: {check_message(fake_attack)}")


#sav
# Save the model weights
torch.save(model.state_dict(), 'can_autoencoder_model.pth')

# Save the scaler (Crucial! You need this to scale new data the same way)
import joblib
joblib.dump(scaler, 'can_scaler.pkl')

print("\nModel and Scaler saved successfully! You can now use them for real-time detection.")
