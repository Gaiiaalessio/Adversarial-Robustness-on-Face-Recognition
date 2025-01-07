from skimage.io import imread
import numpy as np
import os
import torch

def read_pair(path, device, model=None, return_feat=False):
    """
    Read image and get its logits
    """
    # Controlla se il file esiste
    if not os.path.exists(path):
        print(f"[WARNING] Immagine non trovata: {path}. Saltando...")
        return None  # Ritorna None se il file non esiste

    try:
        img = imread(path).astype(np.float32)
        img = torch.Tensor(img.transpose((2, 0, 1))[None, :]).to(device)
    except Exception as e:
        print(f"[WARNING] Errore durante la lettura dell'immagine: {path}. Errore: {e}")
        return None  # Ritorna None se c'è un errore nella lettura

    if not return_feat:
        return img

    try:
        feat = model.forward(img).detach().requires_grad_(False)
    except Exception as e:
        print(f"[WARNING] Errore durante il calcolo delle feature: {path}. Errore: {e}")
        return img, None  # Ritorna img, None se non riesce a calcolare le feature

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
            minibatches_pair = self.pairs[self.pos:self.pos + self.batch_size]
            self.pos += self.batch_size
            xs, ys, ys_feat = [], [], []
            for pair in minibatches_pair:
                path_src, path_dst = pair

                # Carica immagine sorgente
                img_src = read_pair(path_src, self.device)
                if img_src is None:  # Salta se l'immagine è mancante
                    print(f"[WARNING] Immagine sorgente non valida: {path_src}. Saltando...")
                    continue

                # Carica immagine destinazione
                result = read_pair(path_dst, self.device, self.model, return_feat=True)
                if result is None:  # Salta se l'immagine è mancante
                    print(f"[WARNING] Immagine destinazione non valida: {path_dst}. Saltando...")
                    continue

                img_dst, feat_dst = result
                if feat_dst is None:  # Salta se le feature non possono essere calcolate
                    print(f"[WARNING] Feature non valide per: {path_dst}. Saltando...")
                    continue

                xs.append(img_src)
                ys.append(img_dst)
                ys_feat.append(feat_dst)

            # Controlla se il batch contiene dati validi
            if not xs or not ys or not ys_feat:
                print("[WARNING] Nessun dato valido nel batch. Saltando batch...")
                return self.__next__()

            xs = torch.cat(xs)
            ys = torch.cat(ys)
            ys_feat = torch.cat(ys_feat)
            return xs, ys, ys_feat, minibatches_pair
        else:
            raise StopIteration




