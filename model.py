import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class SegmentationModel(nn.Module):
    """
    Text-conditioned segmentation model using ResNet50 backbone with FPN-like decoder
    """
    def __init__(self, text_hidden_dim=512, pretrained_backbone=True):
        super().__init__()
        
        # Load ResNet50 backbone
        if pretrained_backbone:
            weights = ResNet50_Weights.IMAGENET1K_V2
            resnet = models.resnet50(weights=weights)
        else:
            resnet = models.resnet50(weights=None)
        
        # Extract encoder layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 256 channels, stride 4
        self.layer2 = resnet.layer2  # 512 channels, stride 8
        self.layer3 = resnet.layer3  # 1024 channels, stride 16
        self.layer4 = resnet.layer4  # 2048 channels, stride 32
        
        # Text conditioning - project text embedding to spatial features
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # Decoder with FPN-style architecture
        # Lateral connections
        self.lateral4 = nn.Conv2d(2048, 256, kernel_size=1)
        self.lateral3 = nn.Conv2d(1024, 256, kernel_size=1)
        self.lateral2 = nn.Conv2d(512, 256, kernel_size=1)
        self.lateral1 = nn.Conv2d(256, 256, kernel_size=1)
        
        # Smooth layers
        self.smooth4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.smooth3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.smooth1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        
        # Upsampling layers
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    
    def _add_text_conditioning(self, features, text_emb):
        """
        Add text conditioning to spatial features via FiLM-like modulation
        """
        B, C, H, W = features.shape
        # Project text embedding
        text_feat = self.text_proj(text_emb)  # [B, 256]
        # Reshape and broadcast
        text_feat = text_feat.view(B, 256, 1, 1).expand(B, 256, H, W)
        # Add (could also use multiplication for FiLM)
        return features + text_feat
    
    def forward(self, x, text_emb):
        """
        Forward pass
        Args:
            x: Input images [B, 3, H, W]
            text_emb: Text embeddings [B, text_hidden_dim]
        Returns:
            logits: Segmentation logits [B, 1, H, W]
        """
        input_size = x.shape[-2:]
        
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c1 = self.layer1(x)   # [B, 256, H/4, W/4]
        c2 = self.layer2(c1)  # [B, 512, H/8, W/8]
        c3 = self.layer3(c2)  # [B, 1024, H/16, W/16]
        c4 = self.layer4(c3)  # [B, 2048, H/32, W/32]
        
        # FPN-style decoder with top-down pathway
        # Level 4 (deepest)
        p4 = self.lateral4(c4)  # [B, 256, H/32, W/32]
        p4 = self._add_text_conditioning(p4, text_emb)  # Add text conditioning
        p4 = self.smooth4(p4)
        
        # Level 3
        p3 = self.lateral3(c3)
        p3 = p3 + self.upsample(p4)  # Add upsampled p4
        p3 = self.smooth3(p3)
        
        # Level 2
        p2 = self.lateral2(c2)
        p2 = p2 + self.upsample(p3)
        p2 = self.smooth2(p2)  # [B, 256, H/8, W/8]
        
        # Level 1
        p1 = self.lateral1(c1)
        p1 = p1 + self.upsample(p2)
        p1 = self.smooth1(p1)  # [B, 64, H/4, W/4]
        
        # Upsample to original resolution
        p1 = self.upsample_4x(p1)  # [B, 64, H, W]
        
        # Final segmentation
        logits = self.seg_head(p1)  # [B, 1, H, W]
        
        # Resize to input size if needed
        if logits.shape[-2:] != input_size:
            logits = nn.functional.interpolate(
                logits, size=input_size, mode='bilinear', align_corners=True
            )
        
        return logits


class SimpleSegmentationModel(nn.Module):
    """
    Simpler baseline model with U-Net style architecture
    """
    def __init__(self, text_hidden_dim=512, pretrained_backbone=True):
        super().__init__()
        
        # Encoder (ResNet50)
        if pretrained_backbone:
            weights = ResNet50_Weights.IMAGENET1K_V2
            resnet = models.resnet50(weights=weights)
        else:
            resnet = models.resnet50(weights=None)
        
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool1 = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4
        
        # Text conditioning
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Decoder
        self.upconv5 = nn.ConvTranspose2d(2048 + 512, 1024, 2, stride=2)
        self.decoder5 = self._make_decoder_block(1024 + 1024, 512)
        
        self.upconv4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.decoder4 = self._make_decoder_block(512 + 512, 256)
        
        self.upconv3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.decoder3 = self._make_decoder_block(256 + 256, 128)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder2 = self._make_decoder_block(64 + 64, 64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.final = nn.Conv2d(32, 1, 1)
    
    def _make_decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, text_emb):
        # Encoder
        e1 = self.encoder1(x)     # 64, H/2, W/2
        e1_pool = self.pool1(e1)  # 64, H/4, W/4
        e2 = self.encoder2(e1_pool)  # 256, H/4, W/4
        e3 = self.encoder3(e2)    # 512, H/8, W/8
        e4 = self.encoder4(e3)    # 1024, H/16, W/16
        e5 = self.encoder5(e4)    # 2048, H/32, W/32
        
        # Add text conditioning to bottleneck
        text_feat = self.text_proj(text_emb)  # [B, 512]
        B, _, H, W = e5.shape
        text_feat = text_feat.view(B, 512, 1, 1).expand(B, 512, H, W)
        e5 = torch.cat([e5, text_feat], dim=1)  # Concatenate
        
        # Decoder
        d5 = self.upconv5(e5)
        d5 = torch.cat([d5, e4], dim=1)
        d5 = self.decoder5(d5)
        
        d4 = self.upconv4(d5)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        logits = self.final(d1)
        
        return logits
