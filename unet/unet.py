import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------
# Transformer-based U-Net Architecture
# -------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h*w).permute(0, 2, 1)  # [b, h*w, c]
        
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, h*w, self.heads, c // self.heads).transpose(1, 2), qkv)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = torch.matmul(attn, v).transpose(1, 2).reshape(b, h*w, c)
        x = self.proj(x)
        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.attn = MultiHeadSelfAttention(dim, heads=heads)
        
        self.norm2 = nn.GroupNorm(1, dim)
        mlp_hidden_dim = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(mlp_hidden_dim, dim, 1)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.transformer = TransformerBlock(out_channels)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = F.gelu(self.conv(x))
        x = self.transformer(x)
        return self.pool(x), x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.transformer = TransformerBlock(out_channels)
        
    def forward(self, x, skip):
        x = self.up(x)
        # Handle dimension mismatches between upsampled feature and skip connection
        # This can happen with odd-sized feature maps
        if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
            # Adjust dimensions using interpolation to match skip connection size
            # dataset has dimensions [2, 24, 49, 80] where 49 is an odd number
            # After 3 downsampling operations (MaxPool2d), 49 becomes approximately 6
            # When upsampling with ConvTranspose2d, 6 becomes 12, then 24, then 48, not 49
            x = F.interpolate(x, size=(skip.shape[2], skip.shape[3]), mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = F.gelu(self.conv(x))
        x = self.transformer(x)
        return x

class TransformerUNet(nn.Module):
    def __init__(self, in_channels=2, base_channels=64):
        super().__init__()
        # Initial projection
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        

        # Down path
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 32, kernel_size=3, padding=1),
            TransformerBlock(base_channels * 32),
            nn.Conv2d(base_channels * 32, base_channels * 16, kernel_size=3, padding=1),
        )
        
        # Up path
        self.up1 = UpBlock(base_channels * 16, base_channels * 8)
        self.up2 = UpBlock(base_channels * 8, base_channels * 4)
        self.up3 = UpBlock(base_channels * 4, base_channels)
        
        # Output
        self.output = nn.Conv2d(base_channels, in_channels, kernel_size=1)
        
    def forward(self, x):        
        # Initial features
        x = self.init_conv(x)
        
        # Down path
        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        
        # Bottleneck
        x = self.bottleneck(x3)
        
        # Up path
        x = self.up1(x, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)
        
        # Output
        x = self.output(x)
        
        return x
    
def train_unet(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}")

if __name__ == "__main__":
    # Create a dummy dataset
    class DummyDataset(Dataset):
        def __init__(self, num_samples=100, num_fields=2, height=64, width=64):
            self.num_samples = num_samples
            self.num_fields = num_fields
            self.height = height
            self.width = width
            self.data = torch.randn(num_samples, num_fields, height, width)
            self.targets = torch.randn(num_samples, num_fields, height, width)
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
    
    # Create train and validation data loaders
    train_dataset = DummyDataset(num_samples=1000)
    val_dataset = DummyDataset(num_samples=200)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Initialize model and train
    model = TransformerUNet(in_channels=2, base_channels=64).to(device)
    train_unet(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4)