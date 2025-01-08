import os
import urllib
import torch
import torch.nn as nn
from RobFR.networks.transform import transform_modules 
import errno



def get_model(url, net, device='cuda'):
    model_name = url.split('/')[-1]
    try:
        print(f"[DEBUG] Caricamento checkpoint: ./ckpts/{model_name}")
        checkpoint = torch.load(f'./ckpts/{model_name}', map_location=device)
        print(f"[DEBUG] Checkpoint caricato con successo.")
        print(f"[DEBUG] Struttura del checkpoint: {checkpoint.keys() if isinstance(checkpoint, dict) else 'non-dict checkpoint'}")
    except Exception as e:
        print(f"[ERROR] Problema nel caricamento del checkpoint: {e}")
        raise

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        net.load_state_dict(checkpoint['state_dict'])
    else:
        net.load_state_dict(checkpoint)
    net.eval()
    net = net.to(device)
    return net


class FaceModel(nn.Module):
    def __init__(self, url, net, **kwargs):
        super(FaceModel, self).__init__()
        embedding_size = kwargs.get('embedding_size', 512)
        device = kwargs.get('device', 'cuda')
        # get the pytorch model
        self.net = get_model(
                net=net,
                url=url,
                device=device)
        out_dims = embedding_size
        self.out_dims = out_dims
        self.channel = kwargs.get('channel', 'rgb')
        transform_method = kwargs.get('transform', 'None')
        self.transform_module = transform_modules[transform_method]()
        self.iter = 10 if transform_method == 'Randomization' else 1
    def forward(self, x, use_prelogits=False):
       logits = 0
       x_transform = []
       for i in range(self.iter):
           x_transform.append(self.transform_module(x))
       x_transform = torch.cat(x_transform)
       if self.channel == 'bgr':
           x_transform = torch.flip(x_transform, dims=[1])
       x_transform = self.net(x_transform)
       if not use_prelogits:
            logits = x_transform / torch.sqrt(torch.sum(x_transform**2, dim=1, keepdim=True) + 1e-5)
       logits = logits.view(x.shape[0], self.iter, -1)
       return logits.view(self.iter, x.shape[0], -1).mean(dim=0)
    def zero_grad(self):
        self.net.zero_grad()
    def to(self, device):
        self.net = self.net.to(device)