import numpy as np
import torch
import os
import argparse
from RobFR.networks.get_model import getmodel
from RobFR.networks.config import THRESHOLD_DICT
from RobFR.benchmark.utils import run_white, binsearch_basic
from RobFR.dataset import LOADER_DICT
import RobFR.attack as attack
from RobFR.networks.MobileFace import MobileFacenet
from RobFR.networks.MobileFace import MobileFace

print(f"[DEBUG] Directory di esecuzione: {os.getcwd()}")

# Parser degli argomenti
parser = argparse.ArgumentParser()
parser.add_argument('--device', help='device id', type=str, default='cuda')
parser.add_argument('--dataset', help='dataset', type=str, default='lfw', choices=['lfw', 'ytf', 'cfp'])
parser.add_argument('--model', help='White-box model', type=str, default='MobileFace')
parser.add_argument('--goal', help='dodging or impersonate', type=str, default='impersonate', choices=['dodging', 'impersonate'])
parser.add_argument('--eps', help='epsilon', type=float, default=64)
parser.add_argument('--seed', help='random seed', type=int, default=1234)
parser.add_argument('--batch_size', help='batch_size', type=int, default=20)
parser.add_argument('--steps', help='search steps', type=int, default=5)
parser.add_argument('--bin_steps', help='binary search steps', type=int, default=5)
parser.add_argument('--distance', help='l2 or linf', type=str, default='linf', choices=['linf', 'l2'])
parser.add_argument('--log', help='log file', type=str, default='log.txt')
parser.add_argument('--output', help='output dir', type=str, default='output/expdemo')

args = parser.parse_args()

# Imposta il seme per la riproducibilità
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def main():
    device = args.device
    if args.device == "gpu":
        device = "cuda"
    
    kwargs = {
        'device': device 
    }
    try:
        if args.model == 'MobileFace':
            model = MobileFace(device=device)
            img_shape = (112, 112)
        else:
            raise ValueError(f"Modello `{args.model}` non supportato.")
        if model is None:
            raise ValueError("[ERROR] Il modello non è stato inizializzato correttamente.")
        print(f"[DEBUG] Modello `{args.model}` caricato con successo. Dimensione immagine: {img_shape}")
    except Exception as e:
        raise ValueError(f"[ERROR] Il modello `{args.model}` non è stato caricato correttamente. Errore: {e}")

    config = {
        'eps': args.eps,
        'method': attack.FGSM,
        'goal': args.goal,
        'distance_metric': args.distance,
        'threshold': THRESHOLD_DICT[args.dataset][args.model]['cos'],
        'steps': args.steps,
        'bin_steps': args.bin_steps,
        'model': model,
    }
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    datadir = os.path.join(project_root, 'data/lfw-112x112-new'.format(args.dataset, img_shape[0], img_shape[1]))

    loader = LOADER_DICT[args.dataset](datadir, args.goal, args.batch_size, model)
    Attacker = lambda xs, ys, ys_feat, pairs: binsearch_basic(xs=xs, ys=ys, 
        ys_feat=ys_feat, pairs=pairs,**config)
    
    # Chiama run_white senza parametro device
    run_white(loader, Attacker, model, args)

if __name__ == '__main__':
    main()
