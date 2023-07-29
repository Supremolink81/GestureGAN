from pipeline_setup import *
from gan_pipeline import GANPipeline, weights_init
from generator import ConvolutionalGenerator
from discriminator import Discriminator
import random
import time
from pytorch_fid.fid_score import *

if __name__ == "__main__":

    gpu: torch.device = torch.device("cuda:0")

    print(calculate_fid_given_paths(["data", "preprocessed_data"], batch_size=1, device=gpu, dims=2048))

    exit()

    LATENT_VECTOR_SIZE: int = 100

    COLOR_CHANNELS: int = 3

    FEATURE_MAP_SIZE: int = 128

    gpu: torch.device = torch.device("cuda:0")

    random_seed: float = time.time()

    print(random_seed)

    random.seed(random_seed)

    generator: ConvolutionalGenerator = ConvolutionalGenerator(color_channels=COLOR_CHANNELS, latent_vector_size=LATENT_VECTOR_SIZE, feature_map_size=FEATURE_MAP_SIZE).to(gpu)

    generator.load_state_dict(torch.load("GestureGenerator.pth"))

    for k in range(50):

        latent_vector_tensor_size: tuple[int, int, int, int] = (50, LATENT_VECTOR_SIZE, 1, 1)

        random_tensor: torch.Tensor = torch.randn(latent_vector_tensor_size).to(gpu)

        generated_image_tensor: torch.Tensor = generator(random_tensor)

        generated_image_grid = torchvision.utils.make_grid(generated_image_tensor, normalize=True, nrow=1, padding=0)

        for i in range(50):

            generated_image: Image.Image = preprocessing.tensor_to_image(generated_image_grid[:, 64*i:64*(i+1), :].cpu())

            generated_image.save(f"generated_data/img{50*k+i}.png")

    generator_loss: torch.nn.modules.loss._Loss = torch.nn.BCELoss().to(gpu)

    discriminator_loss: torch.nn.modules.loss._Loss = torch.nn.BCELoss().to(gpu)

    print("Gesture data loading...")
    
    gesture_dataset: GestureGANDataset = load_generator_dataset("data", preload_tensors=True)

    print(f"Gesture data of size {len(gesture_dataset)} loaded.")

    while True:

        LEARNING_RATE, BETA1, BETA2, EPOCHS, BATCH_SIZE = get_training_hyperparameters()

        generator: ConvolutionalGenerator = ConvolutionalGenerator(color_channels=3, latent_vector_size=100, feature_map_size=128).to(gpu)

        generator.apply(weights_init)

        generator_optimizer: torch.optim.Optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
                                                                      
        discriminator: Discriminator = Discriminator(color_channels=3, feature_map_size=64).to(gpu)
        
        discriminator.apply(weights_init)

        discriminator_optimizer: torch.optim.Optimizer = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

        gan_pipeline: GANPipeline = GANPipeline(generator, generator_optimizer, generator_loss, discriminator, discriminator_optimizer, discriminator_loss)

        generator_loss_values, discriminator_loss_values = gan_pipeline.train(gesture_dataset, EPOCHS, BATCH_SIZE, gpu)

        plot_gan_loss_graphs(generator_loss_values, discriminator_loss_values)

        display_gan_results(generator, 100, 64)

        torch.save(generator.state_dict(), "GestureGenerator.pth")