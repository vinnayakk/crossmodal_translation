import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Paths to input feature files
AUDIO_SEGMENTED_PATH = "your path here"
VIDEO_SEGMENTED_PATH = "your path here"
TEXT_SEGMENTED_PATH = "your path here"

# Load features
print("Loading features...")
audio_features = np.load(AUDIO_SEGMENTED_PATH)
video_features = np.load(VIDEO_SEGMENTED_PATH)
text_descriptions = np.load(TEXT_SEGMENTED_PATH, allow_pickle=True)

# Confirm shapes
print(f"Audio Features Shape: {audio_features.shape}")
print(f"Video Features Shape: {video_features.shape}")
print(f"Text Descriptions Shape: {text_descriptions.shape}")

# Inspecting rows
print("Inspecting audio sample...")
print(audio_features[0])

print("Inspecting video sample...")
print(video_features[0])

print(f"Sample Text Description: {text_descriptions[0]}")

"""This section is for Group Based splitting for Symbol Grounding:"""

num_objects = 5
num_actions = 5
num_groups = 36
samples_per_group = 1000
total_samples = num_groups * samples_per_group

# Assign groups and group IDs
groups = []

for group_id in range(num_groups):
  # Static
  for object_id in range(num_objects):
    group_name = f"object{object_id}-still"
    groups.extend([group_name] * 5)

  # In Action
  for object_id in range(num_objects):
    for action_id in range(num_actions):
      group_name = f"object{object_id}-action{action_id}"
      groups.append(group_name)

groups = np.tile(groups, total_samples // len(groups))
groups = np.array(groups)

# Validate
print(f"Length of groups before validation: {len(groups)}")
assert len(groups) == total_samples, f"Expected {total_samples} groups, but got {len(groups)}"
print(f"Validation Successful: {len(groups)} entries.")

# Group based splitter
splitter = GroupShuffleSplit(test_size = 0.2, n_splits = 1, random_state = 42 )
train_indices, test_indices = next(splitter.split(audio_features, groups=groups))

# Test and train split
audio_train, audio_test = audio_features[train_indices], audio_features[test_indices]
video_train, video_test = video_features[train_indices], video_features[test_indices]
text_train, text_test = text_descriptions[train_indices], text_descriptions[test_indices]

# Validation
train_groups = set(groups[train_indices])
test_groups = set(groups[test_indices])

assert train_groups.isdisjoint(test_groups), "Overlap Identified!"
print("Validation Successful")

assert len(audio_train) + len(audio_test) == total_samples, "Train/Test split mismatch!"
print("Validation Successful")

# Save the splits
np.save("your path here", audio_train)
np.save("your path here", audio_test)
np.save("your path here", video_train)
np.save("your path here", video_test)
np.save("your path here", text_train)
np.save("your path here", text_test)

print("The group based splits for training and testing are saved successfully.")

"""This section is for Compositional Semantics Split:


"""

test_object = 0 # The object to be held out

# Fresh train and test indices
train_indices = []
test_indices = []

# Segregation
for i, group in enumerate(groups):
  if f"object{test_object}-action" in group:
    test_indices.append(i)
  else:
    train_indices.append(i)

# Convert to numpy
train_indices = np.array(train_indices)
test_indices = np.array(test_indices)

# Test and train split
audio_train, audio_test = audio_features[train_indices], audio_features[test_indices]
video_train, video_test = video_features[train_indices], video_features[test_indices]
text_train, text_test = text_descriptions[train_indices], text_descriptions[test_indices]

# Validation
test_groups = [group for i, group in enumerate(groups) if i in test_indices]
assert all(f"object{test_object}-action" in g for g in test_groups), "Overlap Identified!"

train_groups = [group for i, group in enumerate(groups) if i in train_indices]
assert all(f"object{test_object}-action" not in g for g in train_groups), "Overlap in training set identified!"

print("Validation Successful")

# Save the splits
np.save("your path here", audio_train)
np.save("your path here", audio_test)
np.save("your path here", video_train)
np.save("your path here", video_test)
np.save("your path here", text_train)
np.save("your path here", text_test)

print("The splits for compositional semantics training and testing are successfully saved.")
