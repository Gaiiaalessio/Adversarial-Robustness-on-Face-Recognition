import os
import torch
import numpy as np
from RobFR.dataset.base import Loader

class LFWLoader(Loader):
    def __init__(self, datadir, goal, batch_size, model):
        super(LFWLoader, self).__init__(batch_size, model)
        
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/pairs_lfw.txt'))
        
        with open(config_path, 'r') as f:
            lines = f.readlines()

        suffix = '.jpg'
        self.pairs = []
        for line in lines:
            line = line.strip().split('\t')
            if len(line) == 3 and goal == 'dodging':
                path_src = os.path.join(datadir, line[0], line[0] + '_' + line[1].zfill(4) + suffix)
                path_dst = os.path.join(datadir, line[0], line[0] + '_' + line[2].zfill(4) + suffix)
            elif len(line) == 4 and goal == 'impersonate':
                path_src = os.path.join(datadir, line[0], line[0] + '_' + line[1].zfill(4) + suffix)
                path_dst = os.path.join(datadir, line[2], line[2] + '_' + line[3].zfill(4) + suffix)
            else:
                continue

            # Controlla se entrambi i file esistono
            if not os.path.exists(path_src) or not os.path.exists(path_dst):
                print(f"[WARNING] Una delle immagini non esiste. Saltando coppia: {path_src}, {path_dst}")
                continue

            self.pairs.append([path_src, path_dst])

