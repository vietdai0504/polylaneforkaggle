import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
from lib.models import PolyRegression, SelfAttention, FeatureFlipBlock  # Import necessary classes

# Define FeatureMapSaver class
class FeatureMapSaver:
    def __init__(self, model):
        self.model = model
        self.feature_maps = OrderedDict()

    def hook_fn(self, module, input, output):
        """Hook function to save Feature Maps."""
        module_name = module.__class__.__name__
        layer_name = f"{module_name}-{id(module)}"
        if layer_name not in self.feature_maps:
            self.feature_maps[layer_name] = []
        
        # Handle different types of outputs
        if isinstance(output, torch.Tensor):
            # Single tensor output
            self.feature_maps[layer_name].append(output.detach().cpu())
        elif isinstance(output, (tuple, list)):
            # Tuple or list output: store all tensors within
            tensors = [o.detach().cpu() for o in output if isinstance(o, torch.Tensor)]
            self.feature_maps[layer_name].extend(tensors)
        else:
            # Unsupported output type
            print(f"Unsupported output type for layer {layer_name}: {type(output)}")

    def register_hooks(self):
        """Register hooks to Conv2d, SelfAttention, and FeatureFlipBlock layers."""
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Conv2d, SelfAttention, FeatureFlipBlock)):
                layer.register_forward_hook(self.hook_fn)
                print(f"Registered hook for {name}: {layer.__class__.__name__}")

def visualize_feature_maps(feature_maps, output_dir="feature_maps"):
    """Visualize and save all Feature Maps."""
    os.makedirs(output_dir, exist_ok=True)
    for layer_name, maps in feature_maps.items():
        for idx, feature_map in enumerate(maps):
            if feature_map.ndimension() == 4:  # Batch x Channels x Height x Width
                num_filters = min(8, feature_map.size(1))  # Show up to 8 channels
                feature_map_to_show = feature_map[0, :num_filters]  # Select first batch

                # Plot feature maps
                fig, axes = plt.subplots(1, num_filters, figsize=(15, 15))
                for j, ax in enumerate(axes):
                    ax.imshow(feature_map_to_show[j].numpy(), cmap='viridis')
                    ax.axis('off')
                plt.tight_layout()
                # Save image
                image_filename = os.path.join(output_dir, f"{layer_name}_feature_map_{idx}.png")
                plt.savefig(image_filename)
                plt.close()
                print(f"Feature Map for {layer_name} saved at {image_filename}")

def preprocess_image(image_path):
    """Preprocess the input image."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def load_model(checkpoint_path):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, weights_only=True)  # Set weights_only=True

    # Create the model
    model = PolyRegression(num_outputs=35, backbone='mobilenet_v2', pretrained=False)

    # Load state_dict
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()  # Switch to evaluation mode
    return model

def main(image_path, output_dir, checkpoint_path):
    # Load model from checkpoint
    model = load_model(checkpoint_path)

    # Create FeatureMapSaver instance and register hooks
    saver = FeatureMapSaver(model)
    saver.register_hooks()

    # Load and preprocess image
    input_tensor = preprocess_image(image_path)

    # Pass through the model
    with torch.no_grad():
        _ = model(input_tensor)

    # Save and visualize Feature Maps
    visualize_feature_maps(saver.feature_maps, output_dir)

if __name__ == "__main__":
    image_path = "0000.png"  # Path to your image
    output_dir = "feature_maps"  # Directory to save Feature Maps
    checkpoint_path = "D:/manga/nckh_polylanenet/experiments/tusimple/models/model_021.pt"  # Path to checkpoint

    main(image_path, output_dir, checkpoint_path)
