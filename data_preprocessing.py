import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

# Paths to input and output files
AUDIO_INPUT_PATH = "/your path goes here"
VIDEO_INPUT_PATH = "/your path goes here"
AUDIO_OUTPUT_PATH = "/your path goes here"
VIDEO_OUTPUT_PATH = "/your path goes here"
TEXT_OUTPUT_PATH = "/your path goes here"

# Load Audio Features
print("Loading audio features...")
audio_features = np.load(AUDIO_INPUT_PATH, allow_pickle=True)

# Parameters
fixed_length = 150

# Step 1: Extract Audio Tokens and Handle Variable Lengths
print("Extracting and processing audio tokens...")
audio_tokens = []
audio_texts = []

for row in audio_features:
    tensor_data = row[0]  # Extract the token tensor
    text_description = row[1] # Extract the text

    numpy_data = tensor_data.cpu().numpy().squeeze()  

    # Pad or truncate to fixed length
    if numpy_data.shape[0] < fixed_length:
      padding_value = np.mean(audio_tokens)
      padded_data = np.pad(numpy_data, (0, fixed_length - numpy_data.shape[0]), mode='constant', constant_values=padding_value)
      padded_data = np.pad(numpy_data, (0, fixed_length - numpy_data.shape[0]), mode='constant')
    else:
      padded_data = numpy_data[:fixed_length]

    audio_tokens.append(padded_data)
    audio_texts.append(text_description)

# Convert the list to a NumPy array
audio_tokens = np.array(audio_tokens)
audio_texts = np.array(audio_texts)
print(f"Audio Tokens Extracted. Shape: {audio_tokens.shape}")
print(f"Text Descriptions Extracted. Shape: {audio_texts.shape}")

# Step 2: Normalize Audio Features
print("Normalizing audio features...")
scaler_audio = MinMaxScaler()
audio_tokens_normalized = scaler_audio.fit_transform(audio_tokens)

# Save Normalized Audio Features
np.save(AUDIO_OUTPUT_PATH, audio_tokens_normalized)
print(f"Audio Features Normalized and Saved to {AUDIO_OUTPUT_PATH}.")

# Save Text Descriptions
np.save(TEXT_OUTPUT_PATH, audio_texts)
print(f"Text Descriptions Saved to {TEXT_OUTPUT_PATH}.")

# Load Video Features
print("Loading video features...")
video_features = np.load(VIDEO_INPUT_PATH)

# Step 3: Normalize Video Features
print("Normalizing video features...")
scaler_video = MinMaxScaler()
video_features_normalized = scaler_video.fit_transform(video_features)

# Save Normalized Video Features
np.save(VIDEO_OUTPUT_PATH, video_features_normalized)
print(f"Video Features Normalized and Saved to {VIDEO_OUTPUT_PATH}.")

# Validation
print("Validation:")
print(f"Audio Normalized Shape: {audio_tokens_normalized.shape}, Min: {audio_tokens_normalized.min()}, Max: {audio_tokens_normalized.max()}")
print(f"Video Normalized Shape: {video_features_normalized.shape}, Min: {video_features_normalized.min()}, Max: {video_features_normalized.max()}")
print(f"Sample Text Description: {audio_texts[0]}")
