import torch
import torch.utils.data as data
import torch.nn as nn

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class GANPipeline:

    generator: torch.nn.Module
    generator_optimizer: torch.optim.Optimizer
    generator_loss: torch.nn.modules.loss._Loss

    discriminator: torch.nn.Module
    discriminator_optimizer: torch.optim.Optimizer
    discriminator_loss: torch.nn.modules.loss._Loss

    def __init__(self, 
        generator: torch.nn.Module, 
        generator_optimizer: torch.optim.Optimizer, 
        generator_loss: torch.nn.modules.loss._Loss,
        discriminator: torch.nn.Module,
        discriminator_optimizer: torch.optim.Optimizer,
        discriminator_loss: torch.nn.modules.loss._Loss,
    ):
        
        self.generator = generator

        self.generator_optimizer = generator_optimizer

        self.generator_loss = generator_loss

        self.discriminator = discriminator

        self.discriminator_optimizer = discriminator_optimizer

        self.discriminator_loss = discriminator_loss

    def train(self, training_data: data.Dataset, epochs: int, batch_size: int = 1, gpu: torch.device = None, learning_rate_scheduler: torch.optim.lr_scheduler._LRScheduler = None) -> tuple[list[float], list[float]]:

        dataloader: data.DataLoader = data.DataLoader(training_data, batch_size, shuffle=True)

        generator_losses: list[float] = []

        discriminator_losses: list[float] = []

        batch_number: int = 0

        for epoch_number in range(epochs):

            for training_batch in dataloader:

                training_batch = training_batch.float()

                training_batch += torch.randn(training_batch.shape)

                if gpu:

                    training_batch = training_batch.to(gpu)

                # real image training

                self.discriminator_optimizer.zero_grad()

                real_batch_labels: torch.Tensor = torch.full((batch_size,), 1.0, device=gpu).float() + 0.05 * torch.randn((batch_size,), device=gpu)

                real_batch_output: torch.Tensor = self.discriminator(training_batch).view(-1)

                real_label_error: torch.Tensor = self.discriminator_loss(real_batch_output, real_batch_labels)

                real_label_error.backward()

                # fake image training

                fake_batch_labels: torch.Tensor = torch.full((batch_size,), 0.0, device=gpu).float() + 0.05 * torch.randn((batch_size,), device=gpu)

                noise_tensor_size: tuple[int, int, int, int] = (batch_size, self.generator.latent_vector_size, 1, 1)

                noise_tensors: torch.Tensor = torch.normal(torch.zeros(noise_tensor_size), torch.ones(noise_tensor_size)).to(gpu)

                fake_images: torch.Tensor = self.generator(noise_tensors)

                # use detach() to ensure we can do gradient pass on discriminator later
                fake_batch_output: torch.Tensor = self.discriminator(fake_images.detach()).view(-1)

                fake_label_error: torch.Tensor = self.discriminator_loss(fake_batch_output, fake_batch_labels)

                fake_label_error.backward()

                discriminator_error: float = fake_label_error.item() + real_label_error.item()

                self.discriminator_optimizer.step()

                discriminator_losses.append(discriminator_error)

                # generator training

                self.generator_optimizer.zero_grad()

                fake_image_discriminator_output: torch.Tensor = self.discriminator(fake_images).view(-1)

                # fake labels are real for generator loss
                generator_error: torch.Tensor = self.generator_loss(fake_image_discriminator_output, real_batch_labels)

                generator_error.backward()

                generator_losses.append(generator_error.item())

                self.generator_optimizer.step()

                print(f"Batch {batch_number+1} done.")

                batch_number += 1

            print(f"Epoch {epoch_number+1} done.")

            if learning_rate_scheduler is not None:

                learning_rate_scheduler.step()
            
        return generator_losses, discriminator_losses