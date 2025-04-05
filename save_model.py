import torch
from main import MolGenerator, MolDiscriminator

def save_model(generator, discriminator, path):
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict()
    }, path)

if __name__ == "__main__":
    # Initialize models with same architecture
    input_dim = 64
    hidden_dim = 128
    generator = MolGenerator(input_dim, hidden_dim, input_dim)
    discriminator = MolDiscriminator(input_dim, hidden_dim)
    
    # Load your existing weights here
    # ... load your weights ...
    
    # Save in the correct format
    save_model(generator, discriminator, 'models/molgan_model.pth')