# Load libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import OneCycleLR

# Class for Beta VAE
class BetaVAE(nn.Module):
  def _compute_flattened_size(self, dummy_input, in_channels, out_channels1, out_channels2):
    conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels1, kernel_size=3, stride=1, padding=1)
    conv2 = nn.Conv1d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=3, stride=1, padding=1)
    pool = nn.MaxPool1d(2)

    x = F.relu(conv1(dummy_input))
    x = F.relu(conv2(x))
    x = pool(x)
    return x.flatten(1).shape[1]

  def __init__(self, z_dim_video = 8, z_dim_audio = 8, beta=0.15, gamma=10.0, max_capacity=13):
    super(BetaVAE, self).__init__()
    self.total_z_dim = z_dim_video + z_dim_audio
    self.beta = beta
    self.gamma = gamma
    self.max_capacity = max_capacity

    # Compute feature sizes
    dummy_video_input = torch.randn(1, 17, 30)
    dummy_audio_input = torch.randn(1, 5, 30)
    video_feature_dim = self._compute_flattened_size(dummy_video_input, 17, 64, 128)
    audio_feature_dim = self._compute_flattened_size(dummy_audio_input, 5, 32, 64)

    # Video Encoder
    self.video_encoder = nn.Sequential(
        nn.Conv1d(in_channels=17, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(video_feature_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 2 * self.total_z_dim)
    )

    # Audio Encoder
    self.audio_encoder = nn.Sequential(
        nn.Conv1d(in_channels=5, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(audio_feature_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 2 * self.total_z_dim) 
    )

    # Latent Mapping for Video
    self.fc_mu_video = nn.Linear(4 * self.total_z_dim, z_dim_video)  
    self.fc_logvar_video = nn.Linear(4 * self.total_z_dim, z_dim_video)

    # Latent Mapping for Audio
    self.fc_mu_audio = nn.Linear(4 * self.total_z_dim, z_dim_audio)  
    self.fc_logvar_audio = nn.Linear(4 * self.total_z_dim, z_dim_audio)

    # Video Decoder
    self.video_decoder = nn.Sequential(
        nn.Linear(z_dim_video, 256),
        nn.ReLU(),
        nn.Linear(256, 128 * 15),
        nn.ReLU(),
        nn.Unflatten(1, (128, 15)),
        nn.Upsample(scale_factor=2, mode='linear'),
        nn.Conv1d(128, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv1d(64, 17, kernel_size=3, padding=1),
        nn.Sigmoid()
    )

    # Audio Decoder
    self.audio_decoder = nn.Sequential(
        nn.Linear(z_dim_audio, 128),
        nn.ReLU(),
        nn.Linear(128, 64 * 15),
        nn.ReLU(),
        nn.Unflatten(1, (64, 15)),
        nn.Upsample(scale_factor=2, mode='linear'),
        nn.Conv1d(64, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv1d(32, 5, kernel_size=3, padding=1),
        nn.Sigmoid()
    )

  # Re-parameterize
  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  def forward(self, video_input, audio_input):
    video_input = video_input.permute(0, 2, 1)
    audio_input = audio_input.permute(0, 2, 1)

    # Encode
    video_encoded = self.video_encoder(video_input)
    audio_encoded = self.audio_encoder(audio_input)
    combined_encoded = torch.cat((video_encoded, audio_encoded), dim = 1)

    # Compute mu and logvar
    mu_video = self.fc_mu_video(combined_encoded)
    logvar_video = self.fc_logvar_video(combined_encoded)
    mu_audio = self.fc_mu_audio(combined_encoded)
    logvar_audio = self.fc_logvar_audio(combined_encoded)

    z_video = self.reparameterize(mu_video, logvar_video)
    z_audio = self.reparameterize(mu_audio, logvar_audio)

    # Permute back
    video_recon = self.video_decoder(z_video).permute(0, 2, 1)
    audio_recon = self.audio_decoder(z_audio).permute(0, 2, 1)

    return video_recon, audio_recon, mu_video, mu_audio, logvar_video, logvar_audio

  def loss_function(self, video_recon, audio_recon, video_target, audio_target, mu_video, mu_audio, logvar_video, logvar_audio, epoch, num_epochs):
    recon_video = F.mse_loss(video_recon, video_target)
    recon_audio = F.mse_loss(audio_recon, audio_target)

    capacity = min(self.max_capacity * epoch / num_epochs, self.max_capacity)

    # Compute KL loss separately for video and audio
    kl_loss_video = -0.5 * torch.mean(1 + logvar_video - mu_video.pow(2) - logvar_video.exp())
    kl_loss_audio = -0.5 * torch.mean(1 + logvar_audio - mu_audio.pow(2) - logvar_audio.exp())

    kl_loss = self.gamma * (torch.abs(kl_loss_video - capacity) + torch.abs(kl_loss_audio - capacity))

    return recon_video + recon_audio + self.beta * kl_loss, recon_video, recon_audio, kl_loss

# Load training data
video_train = np.load("your path here")
audio_train = np.load("your path here")

# Split into training and validation
video_train, video_val, audio_train, audio_val = train_test_split(
    video_train, audio_train, test_size=0.2, random_state=42
)

# Prepare data for training and validation
video_train_tensor = torch.tensor(video_train, dtype=torch.float32)
audio_train_tensor = torch.tensor(audio_train, dtype=torch.float32)
video_val_tensor = torch.tensor(video_val, dtype=torch.float32)
audio_val_tensor = torch.tensor(audio_val, dtype=torch.float32)

# Dataloaders
train_dataset = TensorDataset(video_train_tensor, audio_train_tensor)
val_dataset = TensorDataset(video_val_tensor, audio_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BetaVAE(z_dim_video=8, z_dim_audio=8, beta=0.15, gamma=10, max_capacity=13).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
#scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.8)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=70, eta_min=1e-5)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, steps_per_epoch=len(train_loader), epochs=70)

# Training loop with accuracy tracking
num_epochs = 70
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    recon_loss = 0
    kl_loss = 0
    #model.beta = min(0.05, epoch / 20)
    #model.beta = min(0.2, epoch / 50)
    #model.beta = min(0.07, 0.005 * epoch)
    #model.beta = min(0.07, epoch / 20)
    #model.beta = min(0.07, 0.003 * epoch)
    #model.beta = min(0.07, (epoch / 30) * 0.07)
    model.beta = min(0.07, 0.0025 * epoch)  # last good config
    #model.beta = min(0.1, 0.005 * epoch)
    #model.beta = 0.07 * (1 - np.exp(-epoch / 10))

    # Training phase
    for video, audio in train_loader:
        video, audio = video.to(device), audio.to(device)
        video_recon, audio_recon, mu_video, mu_audio, logvar_video, logvar_audio = model(video, audio)
        loss, r_loss_v, r_loss_a, kl = model.loss_function(video_recon, audio_recon, video, audio, mu_video, mu_audio, logvar_video, logvar_audio, epoch, num_epochs)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        recon_loss += (r_loss_v + r_loss_a).item()
        kl_loss += kl.item()

    # Compute Training Accuracy
    model.eval()
    train_cos_vid, train_cos_aud = 0.0, 0.0
    with torch.no_grad():
        for video_batch, audio_batch in train_loader:
            video_batch, audio_batch = video_batch.to(device), audio_batch.to(device)
            vid_recon, aud_recon, _, _, _, _ = model(video_batch, audio_batch)

            
            vid_flat = video_batch.reshape(video_batch.size(0), -1)
            vid_recon_flat = vid_recon.reshape(vid_recon.size(0), -1)
            train_cos_vid += F.cosine_similarity(vid_flat, vid_recon_flat, dim=1).sum().item()

            aud_flat = audio_batch.reshape(audio_batch.size(0), -1)
            aud_recon_flat = aud_recon.reshape(aud_recon.size(0), -1)
            train_cos_aud += F.cosine_similarity(aud_flat, aud_recon_flat, dim=1).sum().item()

    train_acc = ((train_cos_vid / len(train_dataset)) + (train_cos_aud / len(train_dataset))) / 2
    train_accuracies.append(train_acc)

    # Compute Validation Accuracy
    val_cos_vid, val_cos_aud = 0.0, 0.0
    with torch.no_grad():
        for video_batch, audio_batch in val_loader:
            video_batch, audio_batch = video_batch.to(device), audio_batch.to(device)
            vid_recon, aud_recon, _, _, _, _ = model(video_batch, audio_batch)

            vid_flat = video_batch.reshape(video_batch.size(0), -1)
            vid_recon_flat = vid_recon.reshape(vid_recon.size(0), -1)
            val_cos_vid += F.cosine_similarity(vid_flat, vid_recon_flat, dim=1).sum().item()

            aud_flat = audio_batch.reshape(audio_batch.size(0), -1)
            aud_recon_flat = aud_recon.reshape(aud_recon.size(0), -1)
            val_cos_aud += F.cosine_similarity(aud_flat, aud_recon_flat, dim=1).sum().item()

    val_acc = ((val_cos_vid / len(val_dataset)) + (val_cos_aud / len(val_dataset))) / 2
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}: Loss: {total_loss/len(train_loader):.4f} [Recon: {recon_loss/len(train_loader):.4f}, KL: {kl_loss/len(train_loader):.4f}]")
    print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    #scheduler.step(val_acc)
    #scheduler.step()


# Test Evaluation
video_test = np.load("your path here")
audio_test = np.load("your path here")

test_dataset = TensorDataset(torch.tensor(video_test, dtype=torch.float32).to(device), torch.tensor(audio_test, dtype=torch.float32).to(device))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

test_cos_vid, test_cos_aud = 0.0, 0.0
with torch.no_grad():
    for video_batch, audio_batch in test_loader:
        video_batch, audio_batch = video_batch.to(device), audio_batch.to(device)
        vid_recon, aud_recon, _, _, _, _ = model(video_batch, audio_batch)

        vid_flat = video_batch.reshape(video_batch.size(0), -1)
        vid_recon_flat = vid_recon.reshape(vid_recon.size(0), -1)
        test_cos_vid += F.cosine_similarity(vid_flat, vid_recon_flat, dim=1).sum().item()

        aud_flat = audio_batch.reshape(audio_batch.size(0), -1)
        aud_recon_flat = aud_recon.reshape(aud_recon.size(0), -1)
        test_cos_aud += F.cosine_similarity(aud_flat, aud_recon_flat, dim=1).sum().item()

test_acc = ((test_cos_vid / len(test_dataset)) + (test_cos_aud / len(test_dataset))) / 2
test_cos_dist = ((1 - (test_cos_vid / len(test_dataset))) + (1 - (test_cos_aud / len(test_dataset)))) / 2

# Display results
print(f"\nAverage Training Accuracy: {np.mean(train_accuracies):.4f}")
print(f"Average Validation Accuracy: {np.mean(val_accuracies):.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Cosine Distance: {test_cos_dist:.4f}")

# Save model
torch.save(model.state_dict(), "your path here")
print("Model saved successfully!")

import matplotlib.pyplot as plt

# Plot training and validation accuracy curves
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracies, marker='o', label='Training Accuracy')
plt.plot(epochs, val_accuracies, marker='s', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.legend()
plt.grid(True)
plt.show()
