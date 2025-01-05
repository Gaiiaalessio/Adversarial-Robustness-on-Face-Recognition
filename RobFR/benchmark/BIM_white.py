import numpy as np
import torch
import os
import argparse
from tqdm import tqdm

from RobFR.networks.get_model import getmodel
from RobFR.networks.config import THRESHOLD_DICT
from RobFR.benchmark.utils import binsearch_alpha, run_white
from RobFR.dataset import LOADER_DICT
import RobFR.attack as attack

def setup_device(device_arg):
    """
    Setup and validate the computation device.
    Default to 'cuda' if available, else 'cpu'.
    """
    if device_arg not in ['cuda', 'cpu']:
        device_arg = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device_arg)


def validate_arguments(args):
    """
    Validate command-line arguments to ensure compatibility and logical correctness.
    """
    if not torch.cuda.is_available() and args.device == 'cuda':
        print("Warning: CUDA is not available. Switching to CPU.")
        args.device = 'cpu'


# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--device', help='Device id', type=str, default='cuda')
parser.add_argument('--dataset', help='Dataset', type=str, default='lfw', choices=['lfw', 'ytf', 'cfp'])
parser.add_argument('--model', help='White-box model', type=str, default='MobileFace')
parser.add_argument('--goal', help='Attack goal (dodging/impersonate)', type=str, default='impersonate', choices=['dodging', 'impersonate'])
parser.add_argument('--eps', help='Epsilon for attack', type=float, default=16)
parser.add_argument('--iters', help='Attack iterations', type=int, default=20)
parser.add_argument('--seed', help='Random seed', type=int, default=1234)
parser.add_argument('--steps', help='Number of search steps', type=int, default=5)
parser.add_argument('--bin_steps', help='Number of binary search steps', type=int, default=10)
parser.add_argument('--batch_size', help='Batch size', type=int, default=20)
parser.add_argument('--distance', help='Distance metric (l2/linf)', type=str, default='linf', choices=['linf', 'l2'])
parser.add_argument('--output', help='Output directory', type=str, default='output/expdemo')
parser.add_argument('--log', help='Log file name', type=str, default='log.txt')
args = parser.parse_args()

# Validate arguments
validate_arguments(args)

# Setup random seed
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# Setup device
device = setup_device(args.device)

def main():
    """
    Main function to run the attack.
    """
    try:
        # Load the model and ensure it is moved to the correct device
        model, img_shape = getmodel(args.model, device=device)

        # Attack configuration
        config = {
            'eps': args.eps,
            'method': attack.BIM,
            'goal': args.goal,
            'distance_metric': args.distance,
            'threshold': THRESHOLD_DICT[args.dataset][args.model]['cos'],
            'steps': args.steps,
            'bin_steps': args.bin_steps,
            'model': model,
            'iters': args.iters,
        }

        # Prepare the dataset
        datadir = os.path.join('data', f'{args.dataset}-{img_shape[0]}x{img_shape[1]}')
        if not os.path.exists(datadir):
            raise FileNotFoundError(f"Dataset directory '{datadir}' not found.")

        loader = LOADER_DICT[args.dataset](datadir, args.goal, args.batch_size, model)

        # Define the attacker
        Attacker = lambda xs, ys, ys_feat, pairs: binsearch_alpha(xs=xs, ys=ys,
                                                                  ys_feat=ys_feat, pairs=pairs, **config)

        # Run the white-box attack
        run_white(loader, Attacker, model, args)
        print("Attack completed successfully.")

    except Exception as e:
        print(f"Error during execution: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
