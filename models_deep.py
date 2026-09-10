import torch
import torch.nn as nn
import torch.nn.functional as F

class HeteroscedasticLoss(nn.Module):
    """
    Negative Log Likelihood loss for heteroscedastic aleatoric uncertainty.
    The network must output both mean and log_variance.
    Loss = 0.5 * exp(-log_var) * (true - mean)^2 + 0.5 * log_var
    """
    def __init__(self):
        super().__init__()

    def forwardself, y_pred_mean, y_pred_logvar, y_true):
        precision = torch.exp(-y_pred_logvar)
        loss = 0.5 * precision * (y_true - y_pred_mean)**2 + 0.5 * y_pred_logvar
        return loss.mean()

class TemperatureScaling(nn.Module):
    """
    Calibrates softmax probabilities via temperature scaling.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

class SequenceFaciesModel(nn.Module):
    """
    1D-CNN to capture vertical bedding continuity along depth.
    Input shape: (Batch, Features, Sequence_Length)
    Output shape: (Batch, Num_Classes, Sequence_Length)
    """
    def __init__(self, num_features, num_classes=12):
        super().__init__()
        self.conv1 = nn.Conv1d(num_features, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, padding=2, dilation=2) # Dilated for larger receptive field
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.classifier = nn.Conv1d(128, num_classes, kernel_size=1)
        self.temp_scale = TemperatureScaling()
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        logits = self.classifier(x)
        return self.temp_scale(logits)

class PropertyRegressionHead(nn.Module):
    """
    Shared trunk with multi-task regression heads for physical properties.
    Outputs mean and log_var for heteroscedastic uncertainty.
    """
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.mean_head = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.logvar_head = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        mean = self.mean_head(x)
        logvar = self.logvar_head(x)
        return mean, logvar

class WellLogMultiTaskModel(nn.Module):
    """
    Combined model for Facies Classification and Property Prediction.
    """
    def __init__(self, num_features, num_classes=12):
        super().__init__()
        self.shared_conv = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Facies Branch
        self.facies_branch = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, num_classes, kernel_size=1)
        )
        self.temp_scale = TemperatureScaling()
        
        # Property Branches (phi, Sw, k, Pp)
        self.phi_head = PropertyRegressionHead(128)
        self.sw_head = PropertyRegressionHead(128)
        self.k_head = PropertyRegressionHead(128)
        self.pp_head = PropertyRegressionHead(128)
        
    def forward(self, x):
        feat = self.shared_conv(x)
        
        logits = self.facies_branch(feat)
        calibrated_logits = self.temp_scale(logits)
        
        phi_mean, phi_logvar = self.phi_head(feat)
        sw_mean, sw_logvar = self.sw_head(feat)
        k_mean, k_logvar = self.k_head(feat)
        pp_mean, pp_logvar = self.pp_head(feat)
        
        return {
            "facies_logits": calibrated_logits,
            "phi": (phi_mean, phi_logvar),
            "sw": (sw_mean, sw_logvar),
            "k": (k_mean, k_logvar),
            "pp": (pp_mean, pp_logvar)
        }

class SeismicFaciesModel(nn.Module):
    """
    2D CNN for seismic facies segmentation/classification (Phase 0 Pivot).
    Input: (Batch, Channels=5-angles, Time=1000, Trace=70)
    Output: (Batch, Num_Classes, Time=1000, Trace=70)
    """
    def __init__(self, in_channels=5, num_classes=12):
        super().__init__()
        # Simple U-Net style or fully convolutional encoder-decoder
        self.enc1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.dec1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)
        
        self.temp_scale = TemperatureScaling()
        
    def forward(self, x):
        # x is (B, 5, 1000, 70)
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        
        d1 = F.relu(self.dec1(e2))
        logits = self.classifier(d1)
        
        return self.temp_scale(logits)
