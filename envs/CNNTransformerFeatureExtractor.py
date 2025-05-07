import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNTransformerFeatureExtractor(nn.Module):
    def __init__(self, img_channels=1, cnn_out_channels=32, transformer_embed_dim=128,
                 num_transformer_layers=2, num_heads=4):
        super(CNNTransformerFeatureExtractor, self).__init__()

        # CNN层
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, cnn_out_channels, kernel_size=3, padding=1),  # [B, C, 100, 100] -> [B, 32, 100, 100]
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [B, 32, 50, 50]
            nn.Conv2d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1),  # -> [B, 32, 50, 50]
            nn.ReLU(),
            nn.MaxPool2d(2)  # -> [B, 32, 25, 25]
        )

        # 把 CNN 特征展开为 Transformer 的序列输入
        self.flatten_hw = lambda x: x.flatten(2).transpose(1, 2)  # [B, C, H, W] -> [B, N, C]
        self.linear_proj = nn.Linear(cnn_out_channels, transformer_embed_dim)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # 池化输出为单一向量
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # 自动处理输入维度：支持 [H, W] 或 [B, H, W]
        if x.dim() == 2:  # [100, 100]
            x = x.unsqueeze(0).unsqueeze(0)  # -> [1, 1, 100, 100]
        elif x.dim() == 3:  # [B, 100, 100]
            x = x.unsqueeze(1)  # -> [B, 1, 100, 100]

        x = self.conv(x)                     # -> [B, C, H', W']
        x = self.flatten_hw(x)              # -> [B, N, C]
        x = self.linear_proj(x)             # -> [B, N, d]
        x = self.transformer(x)             # -> [B, N, d]
        x = x.transpose(1, 2)               # -> [B, d, N]
        x = self.pool(x).squeeze(-1)        # -> [B, d]
        return x

if __name__ == "__main__":
    gdop_map = torch.randn(100, 100)  # 单张GDOP热力图

    feature_extractor = CNNTransformerFeatureExtractor()
    features = feature_extractor(gdop_map)  # shape: [1, 128]
    print("Feature vector shape:", features.shape)
