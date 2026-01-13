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
df = pd.read_csv(file_path)


def hex_to_int(val):
    try:
        return int(str(val), 16)
    except:
        return 0


# Convert Hex to Numeric
data_split = df['Data'].str.split(' ', expand=True)
numeric_features = data_split.map(hex_to_int).values

# Scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_features)


# --- NEW FOR LSTM: Create Sequences (Sliding Window) ---
# We group messages into sequences of 10.
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)


SEQUENCE_LENGTH = 10
print(f"Creating sequences of length {SEQUENCE_LENGTH}...")
X_seq = create_sequences(scaled_data, SEQUENCE_LENGTH)
train_tensor = torch.tensor(X_seq, dtype=torch.float32)


# ==========================================
# 2. DEFINE THE LSTM AUTOENCODER
# ==========================================
class LSTM_Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTM_Autoencoder, self).__init__()
        # Encoder: Compresses the sequence
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # Decoder: Reconstructs the sequence
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        self.output_layer = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, 8)
        _, (hidden, _) = self.encoder(x)
        # Hidden state contains the "summary" of the sequence

        # Repeat the hidden state for the decoder
        # (Simplified for example)
        batch_size = x.size(0)
        seq_len = x.size(1)

        # We try to reconstruct the same sequence
        x_recon, _ = self.decoder(hidden.repeat(1, seq_len, 1).view(batch_size, seq_len, -1))
        return self.output_layer(x_recon)


# Initialize Model
# input_dim = 8 bytes, hidden_dim = 16 (the "compressed" representation)
model = LSTM_Autoencoder(input_dim=8, hidden_dim=16)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 3. TRAINING
# ==========================================
print("\nStarting LSTM Training (This takes longer than Dense Autoencoder)...")
epochs = 100  # LSTMs converge faster in epochs but take longer per epoch
batch_size = 64

for epoch in range(epochs):
    # Using small batches because LSTMs are memory intensive
    permutation = torch.randperm(train_tensor.size(0))
    epoch_loss = 0

    # Simple training loop for demonstration
    # In a real project, use a DataLoader
    inputs = train_tensor[permutation[:1000]]  # Training on a subset for speed

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, inputs)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")


# ==========================================
# 4. ANOMALY DETECTION LOGIC
# ==========================================
def detect_sequence_anomaly(new_data_csv):
    """
    To detect an anomaly in LSTM, we need a sequence of 10 messages.
    If the sequence is broken (e.g. an injected message), the error spikes.
    """
    model.eval()
    # ... (Same loading/scaling logic as before) ...
    print("\nScanning file for sequence anomalies...")
    # If reconstruction error for the whole sequence > Threshold:
    # return "Sequence Anomaly Detected"


print("\nLSTM Model Ready.")

# Save the model weights
torch.save(model.state_dict(), 'can_LSTM_model.pth')

# Save the scaler (Crucial! You need this to scale new data the same way)
import joblib
joblib.dump(scaler, 'can_scalerLSTM.pkl')

print("\nModel and Scaler saved successfully! You can now use them for real-time detection.")
