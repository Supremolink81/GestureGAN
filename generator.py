import torch
import torch.nn as nn

class Generator(nn.Module):

    latent_vector_size: int
    
    def __init__(self, latent_vector_size: int):

        super(Generator, self).__init__()

        self.latent_vector_size = latent_vector_size

    def forward(self, latent_vector: torch.Tensor) -> torch.Tensor:

        assert False
    
class ConvolutionalGenerator(Generator):

    color_channels: int

    feature_map_size: int

    def __init__(self, color_channels: int, latent_vector_size: int, feature_map_size: int):

        super(ConvolutionalGenerator, self).__init__(latent_vector_size)

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.color_channels = color_channels

        self.feature_map_size = feature_map_size

        self.latent_to_feature_mapping_8 = nn.Sequential(
            nn.ConvTranspose2d(latent_vector_size, feature_map_size * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            self.leaky_relu,
        )

        self.feature_mapping_8_to_feature_mapping_4 = nn.Sequential(
            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            self.leaky_relu,
        )

        self.feature_mapping_4_to_feature_mapping_2 = nn.Sequential(
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            self.leaky_relu,
        )

        self.feature_mapping_2_to_feature_mapping = nn.Sequential(
            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            self.leaky_relu,
        )

        self.feature_mapping_to_rgb = nn.Sequential(
            nn.ConvTranspose2d( feature_map_size, color_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, latent_vector: torch.Tensor) -> torch.Tensor:

        feature_mapping_8: torch.Tensor = self.latent_to_feature_mapping_8(latent_vector)

        feature_mapping_4: torch.Tensor = self.feature_mapping_8_to_feature_mapping_4(feature_mapping_8)

        feature_mapping_2: torch.Tensor = self.feature_mapping_4_to_feature_mapping_2(feature_mapping_4)

        feature_mapping: torch.Tensor = self.feature_mapping_2_to_feature_mapping(feature_mapping_2)

        return self.feature_mapping_to_rgb(feature_mapping)