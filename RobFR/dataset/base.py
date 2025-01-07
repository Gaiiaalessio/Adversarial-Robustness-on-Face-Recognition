from skimage.io import imread
import numpy as np
import os
import torch

def read_pair(path, device, model=None, return_feat=False):
    if not os.path.exists(path):
        print(f"[WARNING] Immagine non trovata: {path}. Saltando...")
        return None

    try:
        img = imread(path).astype(np.float32)
        
        # Controllo se l'immagine ha 3 canali (RGB)
        if len(img.shape) != 3 or img.shape[2] != 3:
            print(f"[WARNING] Immagine non valida (canali errati): {path}. Saltando...")
            return None

        img = torch.Tensor(img.transpose((2, 0, 1))[None, :]).to(device)
    except Exception as e:
        print(f"[WARNING] Errore durante la lettura dell'immagine: {path}. Errore: {e}")
        return None

    if not return_feat:
        return img

    try:
        feat = model.forward(img).detach().requires_grad_(False)
    except Exception as e:
        print(f"[WARNING] Errore durante il calcolo delle feature: {path}. Errore: {e}")
        return img, None

    return img, feat


class Loader:
    def __init__(self, batch_size, model):
        self.batch_size = batch_size
        self.model = model
        self.device = next(model.parameters()).device
        self.pairs = []
        self.pos = 0
    def __len__(self):
        return len(self.pairs) // self.batch_size
    def __iter__(self):
        return self
    def __next__(self):
        if self.pos < len(self.pairs):
            minibatches_pair = self.pairs[self.pos:self.pos+self.batch_size]
            self.pos += self.batch_size
            xs, ys, ys_feat = [], [], []
            for pair in minibatches_pair:
                path_src, path_dst = pair

                # Debug per verificare i percorsi
                if not os.path.exists(path_src):
                    print(f"[DEBUG] Immagine sorgente non trovata: {path_src}")
                if not os.path.exists(path_dst):
                    print(f"[DEBUG] Immagine destinazione non trovata: {path_dst}")

                img_src = read_pair(path_src, self.device)
                if img_src is None:
                    continue

                result = read_pair(path_dst, self.device, self.model, return_feat=True)
                if result is None:
                    continue

                img_dst, feat_dst = result
                if feat_dst is None:
                    continue

                xs.append(img_src)
                ys.append(img_dst)
                ys_feat.append(feat_dst)

            if not xs or not ys or not ys_feat:
                print("[WARNING] Nessun dato valido nel batch. Saltando batch...")
                return self.__next__()

            xs = torch.cat(xs)
            ys = torch.cat(ys)
            ys_feat = torch.cat(ys_feat)
            return xs, ys, ys_feat, minibatches_pair
        else:
            raise StopIteration


