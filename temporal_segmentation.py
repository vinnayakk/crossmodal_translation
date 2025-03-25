import numpy as np

# Paths to input and output files
AUDIO_INPUT_PATH = "your path here"
VIDEO_INPUT_PATH = "your path here"
TEXT_INPUT_PATH = "your path here"

AUDIO_SEGMENTED_OUTPUT = "your path here"
VIDEO_SEGMENTED_OUTPUT = "your path here"
TEXT_SEGMENTED_OUTPUT = "your path here"

# Load normalized features for audio and video
print("Loading normalized features...")
audio_features = np.load(AUDIO_INPUT_PATH, allow_pickle=True)
video_features = np.load(VIDEO_INPUT_PATH)
text_descriptions = np.load(TEXT_INPUT_PATH, allow_pickle=True)

print(f"Loaded Audio Shape: {audio_features.shape}")
print(f"Loaded Video Shape: {video_features.shape}")
print(f"Loaded Text Shape: {text_descriptions.shape}")

# Parameters
n_time_steps = 30 # 0.1s windows
audio_segment_size = audio_features.shape[1] // n_time_steps
video_segment_size = video_features.shape[1] // n_time_steps
validate_audio_length = audio_segment_size * n_time_steps
validate_video_length = video_segment_size * n_time_steps

# Function to segment features with truncation
def segment_features(features, segment_size, n_time_steps, validate_length):
  segmented = []
  for sample in features:
    truncated_sample = sample[:validate_length]
    frames = truncated_sample.reshape(n_time_steps, segment_size)
    segmented.append(frames)
  return np.array(segmented)

print("Segmenting audio features...")
audio_segmented = segment_features(audio_features, audio_segment_size, n_time_steps, validate_audio_length)
print(f"Segmented audio features shape: {audio_segmented.shape}")

print("Segmenting video features...")
video_segmented = segment_features(video_features, video_segment_size, n_time_steps, validate_video_length)
print(f"Segmented video features shape: {video_segmented.shape}")

if len(text_descriptions) == len(audio_features):
    text_segmented = text_descriptions
    print(f"Segmented text descriptions shape: {text_segmented.shape}")
else:
    raise ValueError("Mismatch between text descriptions and audio samples! Check data alignment.")

# Save the new segmented features
np.save(AUDIO_SEGMENTED_OUTPUT, audio_segmented)
np.save(VIDEO_SEGMENTED_OUTPUT, video_segmented)
np.save(TEXT_SEGMENTED_OUTPUT, text_segmented)
print(f"Segmented features saved: \n- Audio: {AUDIO_SEGMENTED_OUTPUT}\n- Video: {VIDEO_SEGMENTED_OUTPUT}\n- Text: {TEXT_SEGMENTED_OUTPUT}")

# Check shapes for validation
print(f"Audio Segmented Shape: {audio_segmented.shape}")
print(f"Video Segmented Shape: {video_segmented.shape}")
print(f"Text Descriptions Shape: {text_segmented.shape}")
print(f"Sample Text Description: {text_segmented[0]}")
