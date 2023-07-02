import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self, color_channels: int, feature_map_size: int):

        super(Discriminator, self).__init__()

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.rgb_to_feature_map = nn.Sequential(
            nn.Conv2d(color_channels, feature_map_size, kernel_size=4, stride=2, padding=1, bias=False),
            self.leaky_relu,
            nn.Dropout(p=0.2),
        )

        self.feature_map_to_feature_map_2 = nn.Sequential(
            nn.Conv2d(feature_map_size, feature_map_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            self.leaky_relu,
            nn.Dropout(p=0.2),
        )

        self.feature_map_2_to_feature_map_4 = nn.Sequential(
            nn.Conv2d(feature_map_size * 2, feature_map_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            self.leaky_relu,
            nn.Dropout(p=0.2),
        )

        self.feature_map_4_to_feature_map_8 = nn.Sequential(
            nn.Conv2d(feature_map_size * 4, feature_map_size * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            self.leaky_relu,
            nn.Dropout(p=0.2),
        )

        self.feature_map_8_to_prediction = nn.Sequential(
            nn.Conv2d(feature_map_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input_image: torch.Tensor):

        feature_map: torch.Tensor = self.rgb_to_feature_map(input_image)

        feature_map_2: torch.Tensor = self.feature_map_to_feature_map_2(feature_map)

        feature_map_4: torch.Tensor = self.feature_map_2_to_feature_map_4(feature_map_2)

        feature_map_8: torch.Tensor = self.feature_map_4_to_feature_map_8(feature_map_4)

        return self.feature_map_8_to_prediction(feature_map_8)