import numpy as np
import torch
import os
import argparse

from RobFR.networks.get_model import getmodel
from RobFR.networks.config import THRESHOLD_DICT
from RobFR.benchmark.utils import run_white, binsearch_basic
from RobFR.dataset import LOADER_DICT
import RobFR.attack as attack

# Parser degli argomenti
parser = argparse.ArgumentParser()
parser.add_argument('--device', help='device id', type=str, default='cuda')
parser.add_argument('--dataset', help='dataset', type=str, default='lfw', choices=['lfw', 'ytf', 'cfp'])
parser.add_argument('--model', help='White-box model', type=str, default='MobileFace')
parser.add_argument('--goal', help='dodging or impersonate', type=str, default='impersonate', choices=['dodging', 'impersonate'])
parser.add_argument('--eps', help='epsilon', type=float, default=16)
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
    print("Caricamento del modello...")
    model, img_shape = getmodel(args.model, device=args.device)

    # Verifica che il modello sia stato caricato correttamente
    if model is None:
        raise ValueError(f"Errore nel caricamento del modello: {args.model}")

    print("Configurazione del modello completata.")
    print(f"Modello: {args.model}, Dataset: {args.dataset}, Goal: {args.goal}, Epsilon: {args.eps}")

    # Configurazione degli attacchi
    config = {
        'eps': args.eps,
        'method': attack.FGSM,
        'goal': args.goal,
        'distance_metric': args.distance,
        'threshold': THRESHOLD_DICT.get(args.dataset, {}).get(args.model, {}).get('cos', None),
        'steps': args.steps,
        'bin_steps': args.bin_steps,
        'model': model,
    }

    # Verifica che la soglia sia definita
    if config['threshold'] is None:
        raise KeyError(f"Soglia non trovata per dataset={args.dataset} e modello={args.model}")

    # Caricamento del dataset
    datadir = os.path.join('data', '{}-{}x{}'.format(args.dataset, img_shape[0], img_shape[1]))
    if not os.path.exists(datadir):
        raise FileNotFoundError(f"Il dataset non è stato trovato nella directory: {datadir}")

    print(f"Caricamento del dataset da: {datadir}")
    loader = LOADER_DICT[args.dataset](datadir, args.goal, args.batch_size, model)

    # Definizione dell'attacco
    Attacker = lambda xs, ys, ys_feat, pairs: binsearch_basic(xs=xs, ys=ys, 
                                                              ys_feat=ys_feat, pairs=pairs, **config)

    print("Esecuzione del metodo run_white...")
    run_white(loader, Attacker, model, args)

if __name__ == '__main__':
    try:
        main()
        print("Esecuzione completata con successo.")
    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")
        raise
