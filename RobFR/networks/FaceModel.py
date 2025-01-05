import os
from six.moves import urllib
import torch
import torch.nn as nn
from RobFR.networks.transform import transform_modules

def get_model(url, net, device):
    """
    Load a model from a checkpoint or download it if not available locally.

    :param url: a string, the URL for the model checkpoint.
    :param net: the backbone model.
    :param device: the device to load the model onto (CPU or CUDA).
    :return: the loaded model.
    """
    model_name = os.path.basename(url)
    checkpoint_path = os.path.join('./ckpts', model_name)

    # Ensure the ckpts directory exists
    os.makedirs('./ckpts/', exist_ok=True)

    try:
        print('Loading existing checkpoint...')
        checkpoint = torch.load(checkpoint_path, map_location=device)  # Ensure map_location is dynamic
    except FileNotFoundError:
        print('Checkpoint not found. Downloading...')
        urllib.request.urlretrieve(url, checkpoint_path)
        print('Download complete.')
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint from {checkpoint_path}: {e}")

    # Load the state_dict into the model
    try:
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            net.load_state_dict(checkpoint['state_dict'])
        else:
            net.load_state_dict(checkpoint)
    except RuntimeError as e:
        raise RuntimeError(f"Error loading model state_dict: {e}. Check compatibility between the model and the checkpoint.")

    net.eval()
    net = net.to(device)  # Ensure the model is moved to the correct device
    return net


class FaceModel(nn.Module):
    def __init__(self, url, net, **kwargs):
        """
        Initialize the FaceModel.

        :param url: URL for the model checkpoint.
        :param net: the backbone model.
        :param kwargs: additional arguments (e.g., embedding_size, device, channel).
        """
        super(FaceModel, self).__init__()
        self.embedding_size = kwargs.get('embedding_size', 512)
        self.device = kwargs.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.channel = kwargs.get('channel', 'rgb')

        # Load the model onto the specified device
        self.net = get_model(net=net, url=url, device=self.device)

        # Set up transformations
        transform_method = kwargs.get('transform', 'None')
        self.transform_module = transform_modules[transform_method]()
        self.iter = 10 if transform_method == 'Randomization' else 1

    def forward(self, x, use_prelogits=False):
        """
        Forward pass for the FaceModel.

        :param x: input tensor.
        :param use_prelogits: whether to return raw logits instead of normalized features.
        :return: output tensor after processing.
        """
        x = x.to(self.device)  # Ensure input tensor is on the correct device
        x_transform = [self.transform_module(x) for _ in range(self.iter)]
        x_transform = torch.cat(x_transform).to(self.device)

        if self.channel == 'bgr':
            x_transform = torch.flip(x_transform, dims=[1])  # Flip channels for BGR

        features = self.net(x_transform)

        if not use_prelogits:
            features = features / torch.sqrt(torch.sum(features ** 2, dim=1, keepdim=True) + 1e-5)

        features = features.view(x.shape[0], self.iter, -1)
        return features.view(self.iter, x.shape[0], -1).mean(dim=0)

    def zero_grad(self):
        """Zero out the gradients of the model."""
        self.net.zero_grad()

    def to(self, device):
        """
        Move the model to the specified device.

        :param device: device to move the model to (CPU or CUDA).
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.net = self.net.to(device)
        self.device = device
