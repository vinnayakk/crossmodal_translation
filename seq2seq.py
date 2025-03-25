# Required Libraries
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

# Load the latent representations from Beta VAE model
latent_video_train = np.load("your path here")
latent_audio_train = np.load("your path here")

latent_video_test = np.load("your path here")
latent_audio_test = np.load("your path here")

# Data preparation
video_train, video_val, audio_train, audio_val = train_test_split(latent_video_train, latent_audio_train, test_size=0.2, random_state=42)

# Convert to float32 for TensorFlow compatibility
video_train = video_train.astype(np.float32)
audio_train = audio_train.astype(np.float32)
video_val = video_val.astype(np.float32)
audio_val = audio_val.astype(np.float32)
video_test = latent_video_test.astype(np.float32)
audio_test = latent_audio_test.astype(np.float32)

# Reshape for sequence (adding timestep dimension)
def reshape_data(data):
    return data.reshape(data.shape[0], 1, data.shape[1])

video_train = video_train.reshape(video_train.shape[0], 1, video_train.shape[1])
audio_train = audio_train.reshape(audio_train.shape[0], 1, audio_train.shape[1])
video_val = video_val.reshape(video_val.shape[0], 1, video_val.shape[1])
audio_val = audio_val.reshape(audio_val.shape[0], 1, audio_val.shape[1])
video_test = video_test.reshape(video_test.shape[0], 1, video_test.shape[1])
audio_test = audio_test.reshape(audio_test.shape[0], 1, audio_test.shape[1])

print("Shapes after reshaping:")
print(f"Video Train: {video_train.shape}, Audio Train: {audio_train.shape}")
print(f"Video Val: {video_val.shape}, Audio Val: {audio_val.shape}")
print(f"Video Test: {video_test.shape}, Audio Test: {audio_test.shape}")

# Prepare decoder inputs (zero-initialized) and targets
decoder_inputs_train = np.zeros_like(audio_train)
decoder_inputs_val = np.zeros_like(audio_val)
decoder_inputs_test = np.zeros_like(audio_test)

decoder_inputs_train[:, :-1, :] = audio_train[:, 1:, :]
decoder_inputs_val[:, :-1, :] = audio_val[:, 1:, :]
decoder_inputs_test[:, :-1, :] = audio_test[:, 1:, :]

# Model architecture
latent_dim = 8
input_dim = video_train.shape[2]
output_dim = audio_train.shape[2]

# Encoder
encoder_inputs = keras.Input(shape=(1, input_dim))
encoder_lstm = keras.layers.LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = keras.Input(shape=(1, output_dim))
decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = keras.layers.Dense(output_dim, activation='linear')
decoder_outputs = decoder_dense(decoder_outputs)

# Custom metric: Cosine Similarity
def cosine_similarity(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return K.mean(K.sum(y_true * y_pred, axis=-1))

# Compile model
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=['mae', cosine_similarity])
model.summary()

decoder_target_train = np.copy(audio_train)
decoder_target_val = np.copy(audio_val)

# Model training
history = model.fit([video_train, decoder_inputs_train], decoder_target_train, batch_size=64, epochs=25, validation_data=([video_val, decoder_inputs_val], decoder_target_val))

# Evaluation metrics
def print_metrics(history):
    print("\n=== Training Metrics ===")
    print(f"Final Train Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Train MAE: {history.history['mae'][-1]:.4f}")
    print(f"Final Train Cosine: {history.history['cosine_similarity'][-1]:.4f}")

    print("\n=== Validation Metrics ===")
    print(f"Final Val Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Final Val MAE: {history.history['val_mae'][-1]:.4f}")
    print(f"Final Val Cosine: {history.history['val_cosine_similarity'][-1]:.4f}")

print_metrics(history)

# Test evaluation
decoder_target_test = np.copy(audio_test)
test_results = model.evaluate(
    [video_test, decoder_inputs_test],
    decoder_target_test,
    verbose=0
)
print("\n=== Test Metrics ===")
print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test MAE: {test_results[1]:.4f}")
print(f"Test Cosine: {test_results[2]:.4f}")

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Evolution')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['cosine_similarity'], label='Train Cosine')
plt.plot(history.history['val_cosine_similarity'], label='Val Cosine')
plt.title('Cosine Similarity Evolution')
plt.legend()
plt.show()

# Save model
model.save("your path here")
