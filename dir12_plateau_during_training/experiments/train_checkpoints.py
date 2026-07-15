#!/usr/bin/env python3
"""
dir12 — Train the image-models MNIST MLP and save checkpoints DURING training.

Exact reproduction of the branch config (scripts/image/train_mnist_mlp.py):
depth-4 width-200 ReLU MLP (784->200->200->200->10), 1000-sample MNIST subset,
AdamW(lr=1e-3, wd=0.01), MSE on one-hot targets, batch 200, 100k steps,
init_scale 1.0. seed defines the training subset (torch.randint after
manual_seed), so seed 0 reproduces the existing endpoint's subset exactly.

We additionally snapshot the model state_dict at a fixed log-spaced schedule of
steps so downstream analysis can measure plateau emergence over training.

Usage: python experiments/train_checkpoints.py --seed 0
"""
import argparse, json, os, sys
sys.path.insert(0, '/workspace/mars-plateaus-image')

import torch
from src.mnist import MLP, load_mnist

torch.set_num_threads(2)

CKPT_STEPS = [0, 10, 30, 100, 300, 1000, 3000, 10000, 20000, 30000, 50000, 75000, 100000]
HERE = '/workspace/marsv_agent_haoyang/dir12_plateau_during_training'
DATA = '/workspace/mars-plateaus-image/data/mnist'


@torch.no_grad()
def eval_metrics(model, x, y, batch=2000):
    model.eval()
    correct = 0
    conf_sum = 0.0
    n = len(x)
    for i in range(0, n, batch):
        logits = model(x[i:i + batch])
        prob = torch.softmax(logits, dim=1)
        conf, pred = prob.max(dim=1)
        correct += (pred == y[i:i + batch]).sum().item()
        conf_sum += conf.sum().item()
    model.train()
    return correct / n, conf_sum / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=100_000)
    p.add_argument('--train_points', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--init_scale', type=float, default=1.0)
    p.add_argument('--n_test', type=int, default=2000)
    args = p.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.225)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(args.seed)
    train_x, train_y, test_x, test_y = load_mnist(DATA)
    subset_idx = torch.randint(0, len(train_x), (args.train_points,))
    train_x = train_x[subset_idx].to(device)
    train_y = train_y[subset_idx].to(device)
    test_x, test_y = test_x[:args.n_test].to(device), test_y[:args.n_test].to(device)

    model = MLP(depth=4, width=200, activation='relu').to(device)
    with torch.no_grad():
        for param in model.parameters():
            param.data = args.init_scale * param.data

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()
    one_hots = torch.eye(10, device=device)

    ckpt_dir = os.path.join(HERE, 'results', 'ckpts', f'seed{args.seed}')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_set = set(CKPT_STEPS)

    history = {'step': [], 'train_loss': [], 'train_acc': [], 'test_acc': [],
               'train_conf': [], 'test_conf': []}

    def snapshot(step):
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(state, os.path.join(ckpt_dir, f'step{step}.pt'))

    def log(step):
        tr_acc, tr_conf = eval_metrics(model, train_x, train_y)
        te_acc, te_conf = eval_metrics(model, test_x, test_y)
        with torch.no_grad():
            tl = loss_fn(model(train_x), one_hots[train_y]).item()
        history['step'].append(step)
        history['train_loss'].append(tl)
        history['train_acc'].append(tr_acc)
        history['test_acc'].append(te_acc)
        history['train_conf'].append(tr_conf)
        history['test_conf'].append(te_conf)
        print(f"step {step}: loss={tl:.5f} tr_acc={tr_acc:.3f} te_acc={te_acc:.3f} "
              f"te_conf={te_conf:.3f}", flush=True)

    if 0 in ckpt_set:
        snapshot(0)
    log(0)
    log_steps = set([s for s in CKPT_STEPS if s > 0])
    # dense-ish logging on a log grid for training-dynamics figure
    dense = set(int(x) for x in [2, 5, 20, 50, 200, 500, 2000, 5000, 7000, 15000,
                                 40000, 60000, 90000])
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, args.train_points, (args.batch_size,), device=device)
        logits = model(train_x[idx])
        loss = loss_fn(logits, one_hots[train_y[idx]])
        opt.zero_grad(); loss.backward(); opt.step()
        if step in ckpt_set:
            snapshot(step)
        if step in ckpt_set or step in dense:
            log(step)

    with open(os.path.join(ckpt_dir, 'history.json'), 'w') as f:
        json.dump(history, f)
    manifest = {'seed': args.seed, 'ckpt_steps': CKPT_STEPS,
                'config': vars(args), 'subset_idx': subset_idx.tolist()}
    with open(os.path.join(ckpt_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f)
    print(f"saved {len(CKPT_STEPS)} checkpoints to {ckpt_dir}")


if __name__ == '__main__':
    main()
