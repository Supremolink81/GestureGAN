from pipeline_setup import *
from gan_pipeline import GANPipeline
from generator import ConvolutionalGenerator
from discriminator import Discriminator

if __name__ == "__main__":

    gpu: torch.device = torch.device("cuda:0")

    generator_loss: torch.nn.modules.loss._Loss = torch.nn.BCELoss().to(gpu)

    discriminator_loss: torch.nn.modules.loss._Loss = torch.nn.BCELoss().to(gpu)

    print("Gesture data loading...")
    
    gesture_dataset: GestureGANDataset = load_generator_dataset("data", preload_tensors=True)

    print(f"Gesture data of size {len(gesture_dataset)} loaded.")

    while True:

        LEARNING_RATE, BETA1, BETA2, EPOCHS, BATCH_SIZE = get_training_hyperparameters()

        generator: ConvolutionalGenerator = ConvolutionalGenerator(color_channels=3, latent_vector_size=100, feature_map_size=128).to(gpu)

        generator_optimizer: torch.optim.Optimizer = torch.optim.Adam(generator.parameters(), lr=2*LEARNING_RATE, betas=(BETA1, BETA2))
                                                                      
        discriminator: Discriminator = Discriminator(color_channels=3, feature_map_size=64).to(gpu)
        
        discriminator_optimizer: torch.optim.Optimizer = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

        gan_pipeline: GANPipeline = GANPipeline(generator, generator_optimizer, generator_loss, discriminator, discriminator_optimizer, discriminator_loss)

        generator_loss_values, discriminator_loss_values = gan_pipeline.train(gesture_dataset, EPOCHS, BATCH_SIZE, gpu)

        plot_gan_loss_graphs(generator_loss_values, discriminator_loss_values)

        display_gan_results(generator, 100, 64)