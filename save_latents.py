# Load libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

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

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BetaVAE(z_dim_video=8, z_dim_audio=8, beta=0.15, gamma=10, max_capacity=13).to(device)
model.load_state_dict(torch.load("your path here"))
model.eval()

video_train = np.load("your path here")
audio_train = np.load("your path here")

train_dataset = TensorDataset(torch.tensor(video_train, dtype=torch.float32).to(device), torch.tensor(audio_train, dtype=torch.float32).to(device))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

video_test = np.load("your path here")
audio_test = np.load("your path here")

test_dataset = TensorDataset(torch.tensor(video_test, dtype=torch.float32).to(device), torch.tensor(audio_test, dtype=torch.float32).to(device))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

latent_video_train = []
latent_audio_train = []

with torch.no_grad():
    for video_batch, audio_batch in train_loader:
        _, _, mu_video, mu_audio, _, _ = model(video_batch, audio_batch)

        latent_video_train.append(mu_video.cpu().numpy())
        latent_audio_train.append(mu_audio.cpu().numpy())

latent_video_train = np.concatenate(latent_video_train, axis=0)
latent_audio_train = np.concatenate(latent_audio_train, axis=0)

# Save the training latents
np.save("your path here", latent_video_train)
np.save("your path here", latent_audio_train)

latent_video_test = []
latent_audio_test = []

with torch.no_grad():
    for video_batch, audio_batch in test_loader:
        _, _, mu_video, mu_audio, _, _ = model(video_batch, audio_batch)

        latent_video_test.append(mu_video.cpu().numpy())
        latent_audio_test.append(mu_audio.cpu().numpy())

latent_video_test = np.concatenate(latent_video_test, axis=0)
latent_audio_test = np.concatenate(latent_audio_test, axis=0)

# Save the test latents
np.save("your path here", latent_video_test)
np.save("your path here", latent_audio_test)
