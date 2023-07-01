import torch
import torchvision
import numpy as np
from matplotlib import pyplot as plt
from torch.utils import data
import os
import preprocessing
from generator import ConvolutionalGenerator
    
class GestureGANDataset(data.Dataset):

    image_names: list[str]
    image_tensors: list[torch.Tensor]

    def __init__(self, image_names, preload_tensors: bool = False):

        self.image_names = image_names

        self.image_tensors = None

        if preload_tensors:

            self.image_tensors = list(map(preprocessing.preprocess_image, image_names))

    def __getitem__(self, index: int) -> torch.Tensor:

        if self.image_tensors is not None:

            image_tensor: torch.Tensor= self.image_tensors[index]

        else:

            image_name: str = self.image_names[index]

            image_tensor: torch.Tensor = preprocessing.preprocess_image(image_name)

        return image_tensor
    
    def __len__(self):

        return len(self.image_names)

def load_generator_dataset(path: str, preload_tensors: bool = False) -> GestureGANDataset:

    """
    Loads image data from a given path. The path is expected to be to a folder
    that contains various images.

    Args:

        str path: the path to the folder to load.

        bool preload_tensors: whether to preload the tensor data.

    Returns:

        A GestureGANDataset object containing the image data.
    """

    get_full_image_path: callable = lambda image_name: path + "/" + image_name

    images_in_label_folder: list[str] = list(map(get_full_image_path, os.listdir(path)))

    return GestureGANDataset(images_in_label_folder, preload_tensors=preload_tensors)

def get_training_hyperparameters() -> tuple[float, float, int, int]:

    """
    Helper function to get the learning rate, beta values, epochs and batch size.

    Args:

        None

    Returns:

        A tuple containing the learning rate, beta values, epochs and batch size.
    """

    LEARNING_RATE: float = float(input("Enter learning rate: "))

    BETA1: float = float(input("Enter beta1 value: "))

    BETA2: float = float(input("Enter beta2 value: "))

    EPOCHS: int = int(input("Enter epochs: "))

    BATCH_SIZE: int = int(input("Enter batch size: "))

    return (LEARNING_RATE, BETA1, BETA2, EPOCHS, BATCH_SIZE)

def plot_gan_loss_graphs(generator_loss_values: list[float], discriminator_loss_values: list[float]) -> None:

    """
    Displays 2 loss graphs from 2 lists of loss function values,
    one for the generator and one for the discriminator.

    Args:

        list[float] generator_loss_values: the list of loss values for the generator.

        list[float] generator_loss_values: the list of loss values for the generator.

    Returns:

        None
    """

    _, axis = plt.subplots(1, 2)

    axis[0].plot(list(range(len(generator_loss_values))), generator_loss_values)
    axis[0].set_title("Generator Loss Values")
    axis[0].set_xlabel("Batch Number")
    axis[0].set_ylabel("Loss Value")

    axis[1].plot(list(range(len(discriminator_loss_values))), discriminator_loss_values)
    axis[1].set_title("Discriminator Loss Values")
    axis[1].set_xlabel("Batch Number")
    axis[1].set_ylabel("Loss Value")

    plt.show()

def display_gan_results(generator: ConvolutionalGenerator, latent_vector_size: int, image_count: int) -> None:

    """
    Displays a sample of generated images from a generator trained as part of a GAN pipeline.

    Args:

        ConvolutionalGenerator generator: the GAN to use to generate samples.

        int latent_vector_size: the size of the latent vector for the GAN.

        int image_count: the number of images to generate.
    """

    latent_vector_batch: torch.Tensor = torch.zeros(image_count, latent_vector_size, 1, 1)

    for i in range(image_count):

        latent_vector_tensor_size: tuple[int, int, int] = (latent_vector_size, 1, 1)

        mean_tensor: torch.Tensor = torch.zeros(latent_vector_tensor_size)

        std_tensor: torch.Tensor = torch.ones(latent_vector_tensor_size)

        latent_vector_batch[i] = torch.normal(mean_tensor, std_tensor)

    generated_images: torch.Tensor = generator(latent_vector_batch.to(torch.device("cuda:0")))

    image_grid: torch.Tensor = torchvision.utils.make_grid(generated_images, normalize=True)

    image_plot: plt.AxesImage = plt.imshow(np.transpose(image_grid.cpu(), (1,2,0)))

    plt.show()