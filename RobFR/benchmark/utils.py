import os
from skimage.io import imread, imsave
import numpy as np
import torch
from tqdm import tqdm
from RobFR.networks.config import threshold_lfw, threshold_ytf, threshold_cfp
import RobFR.attack as attack

threshold_dict = {
    'lfw': threshold_lfw,
    'ytf': threshold_ytf,
    'cfp': threshold_cfp
}

def save_images(image, original_image, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image = np.clip(image, 0, 255).astype(np.uint8)
    imsave(os.path.join(output_dir, filename), image)

def cosdistance(x, y, offset=1e-5):
    x = x / (torch.sqrt(torch.sum(x**2)) + offset)
    y = y / (torch.sqrt(torch.sum(y**2)) + offset)
    return torch.sum(x * y)

def L2distance(x, y):
    return torch.sqrt(torch.sum((x - y)**2))

def binsearch_basic(xs, ys, ys_feat, pairs, eps, method, threshold, device, steps=0, bin_steps=0, *args, **kwargs):
    batch_size = xs.size(0)

    lo = torch.zeros(batch_size, 1, 1, 1, device=device)
    hi = lo + eps
    eps_tensor = torch.full((batch_size, 1, 1, 1), eps, device=device)
    xs_results = xs.clone().detach()
    goal = kwargs['goal']
    model = kwargs['model'].to(device)
    found = torch.zeros(batch_size, device=device, dtype=torch.bool)

    # Linear search
    for step in range(2**steps):
        magnitude = (1.0 - float(step) / (2**steps)) * eps_tensor
        kwargs['eps'] = magnitude
        attacker = method(**kwargs)
        xs_adv = attacker.batch_attack(xs=xs, ys=ys, ys_feat=ys_feat, pairs=pairs)
        ys_adv = model(xs_adv)
        similarities = torch.sum(ys_adv * ys_feat, dim=1)

        succ_ = (threshold - similarities > 0) if goal == 'dodging' else (similarities - threshold > 0)
        xs_results[succ_] = xs_adv[succ_]
        hi[succ_] = (1.0 - float(step) / (2**steps)) * eps
        found[succ_] = True

    lo = hi - float(eps) / (2**steps)

    # Binary search
    for _ in range(bin_steps):
        mi = (lo + hi) / 2
        kwargs['eps'] = mi
        attacker = method(**kwargs)
        xs_adv = attacker.batch_attack(xs=xs, ys=ys, ys_feat=ys_feat, pairs=pairs)
        ys_adv = model(xs_adv)
        similarities = torch.sum(ys_adv * ys_feat, dim=1)

        succ_ = (threshold - similarities > 0) if goal == 'dodging' else (similarities - threshold > 0)
        hi[succ_] = mi[succ_]
        lo[~succ_] = mi[~succ_]
        xs_results[succ_] = xs_adv[succ_]
        found[succ_] = True

    return xs_results, found

def binsearch_alpha(xs, ys, ys_feat, pairs, eps, method, threshold, device, steps=0, bin_steps=0, *args, **kwargs):
    batch_size = xs.size(0)

    lo = torch.zeros(batch_size, 1, 1, 1, device=device)
    hi = lo + eps
    xs_results = xs.clone().detach()
    goal = kwargs['goal']
    model = kwargs['model'].to(device)
    found = torch.zeros(batch_size, device=device, dtype=torch.bool)

    # Linear search
    for step in range(steps):
        kwargs['eps'] = hi
        attacker = method(**kwargs)
        xs_adv = attacker.batch_attack(xs=xs, ys=ys, ys_feat=ys_feat, pairs=pairs)
        ys_adv = model(xs_adv)
        similarities = torch.sum(ys_adv * ys_feat, dim=1)

        succ_ = (threshold - similarities > 0) if goal == 'dodging' else (similarities - threshold > 0)
        cond = ~found & succ_
        xs_results[cond] = xs_adv[cond]
        found[cond] = True
        not_found = ~found
        lo[not_found] = hi[not_found]
        hi[not_found] *= 2

        if found.all():
            break

    # Binary search
    for _ in range(bin_steps):
        mi = (lo + hi) / 2
        kwargs['eps'] = mi
        attacker = method(**kwargs)
        xs_adv = attacker.batch_attack(xs=xs, ys=ys, ys_feat=ys_feat, pairs=pairs)
        ys_adv = model(xs_adv)
        similarities = torch.sum(ys_adv * ys_feat, dim=1)

        succ_ = (threshold - similarities > 0) if goal == 'dodging' else (similarities - threshold > 0)
        hi[succ_] = mi[succ_]
        lo[~succ_] = mi[~succ_]
        xs_results[succ_] = xs_adv[succ_]
        found[succ_] = True

    return xs_results, found

def run_black(loader, attacker, args, device):
    os.makedirs(args.output, exist_ok=True)
    cnt = 0
    outputs = []

    for xs, ys, ys_feat, pairs in tqdm(loader, total=len(loader)):
        xs, ys, ys_feat = xs.to(device), ys.to(device), ys_feat.to(device)
        x_adv = attacker.batch_attack(xs=xs, ys=ys, ys_feat=ys_feat, pairs=pairs)

        for i in range(len(pairs)):
            img = x_adv[i].detach().cpu().numpy().transpose((1, 2, 0))
            cnt += 1
            npy_path = os.path.join(args.output, f"{cnt}.npy")
            outputs.append([npy_path, pairs[i][0], pairs[i][1]])
            np.save(npy_path, img)
            original_image = xs[i].cpu().numpy().transpose((1, 2, 0))
            save_images(img, original_image, f"{cnt}.png", args.output)

    with open(os.path.join(args.output, 'annotation.txt'), 'w') as f:
        for pair in outputs:
            f.write(f"{pair[0]} {pair[1]} {pair[2]}\n")

def run_white(loader, attacker, model, args, device):
    os.makedirs('log', exist_ok=True)
    os.makedirs(args.output, exist_ok=True)
    cnt = 0
    scores, dists, success, advs, imgs = [], [], [], [], []

    model = model.to(device)

    for xs, ys, ys_feat, pairs in tqdm(loader, total=len(loader)):
        # Verifica batch vuoti o dati mancanti
        if xs is None or ys is None or len(xs) == 0:
            print(f"[WARNING] Batch vuoto o dati mancanti. Saltando...")
            continue

        try:
            # Passa il batch al dispositivo
            xs, ys, ys_feat = xs.to(device), ys.to(device), ys_feat.to(device)
            
            # Attacco e predizione
            x_adv, found = attacker(xs=xs, ys=ys, ys_feat=ys_feat, pairs=pairs)
            y_adv = model(x_adv)
            s = torch.sum(y_adv * ys_feat, dim=1)

            # Elaborazione dei risultati
            for i in range(len(pairs)):
                img = x_adv[i].detach().cpu().numpy().transpose((1, 2, 0))
                x = xs[i].detach().cpu().numpy().transpose((1, 2, 0))
                scores.append(s[i].item())
                success.append(int(found[i].item()))
                cnt += 1
                advs.append(f"{cnt}.npy")
                npy_path = os.path.join(args.output, f"{cnt}.npy")
                np.save(npy_path, img)

                # Calcola la distanza
                if args.distance == 'l2':
                    dist = np.linalg.norm(img - x) / np.sqrt(img.size)
                else:
                    dist = np.max(np.abs(img - x))

                dists.append(dist)
                imgs.append(pairs[i][1])

        except Exception as e:
            print(f"[ERROR] Errore durante l'elaborazione del batch. Dettagli: {e}")
            continue  # Salta il batch in caso di errore

    # Salvataggio dei risultati nel file di log
    with open(os.path.join('log', args.log), 'w') as f:
        f.write('adv_img,tar_img,score,dist,success\n')
        for adv, img, score, d, s in zip(advs, imgs, scores, dists, success):
            f.write(f"{adv},{img},{score},{d},{s}\n")



