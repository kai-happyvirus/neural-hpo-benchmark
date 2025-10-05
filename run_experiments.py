#!/usr/bin/env python3
"""
Quick experiments runner: GA/DE/PSO on MNIST (PyTorch).
- Keeps settings small for a fast smoke test.
- Logs results to results.csv (best_acc, algo, params).
"""

import time, random, math, csv, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from deap import base, creator, tools, algorithms

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- Data ----------
def load_mnist(batch_size):
    tfm = transforms.ToTensor()
    train = datasets.MNIST(root='./data', train=True, download=True, transform=tfm)
    test  = datasets.MNIST(root='./data', train=False, download=True, transform=tfm)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test,  batch_size=512, shuffle=False)
    return train_loader, test_loader

# ---------- Model ----------
class FNN(nn.Module):
    def __init__(self, hidden=[128, 64], dropout=0.2):
        super().__init__()
        layers = []
        in_dim = 28*28
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, 10)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

# ---------- Search Space ----------
BATCH_CHOICES = [32, 64, 128]
HIDDEN_CHOICES = [[128],[256],[256,128],[512,256]]
OPTIMIZERS = ['sgd','adam','rmsprop']
LR_MIN, LR_MAX = 1e-4, 1e-1
DROP_MIN, DROP_MAX = 0.1, 0.5

def decode(ind):
    lr = 10 ** (np.log10(LR_MIN) + ind[0]*(np.log10(LR_MAX)-np.log10(LR_MIN)))
    bsz = BATCH_CHOICES[int(ind[1]*len(BATCH_CHOICES)) % len(BATCH_CHOICES)]
    drp = DROP_MIN + ind[2]*(DROP_MAX-DROP_MIN)
    hid = HIDDEN_CHOICES[int(ind[3]*len(HIDDEN_CHOICES)) % len(HIDDEN_CHOICES)]
    opt = OPTIMIZERS[int(ind[4]*len(OPTIMIZERS)) % len(OPTIMIZERS)]
    return {'lr':float(lr),'batch':int(bsz),'dropout':float(drp),'hidden':hid,'opt':opt}

def get_optimizer(name, params, lr):
    if name=='sgd': return optim.SGD(params, lr=lr, momentum=0.9)
    if name=='adam': return optim.Adam(params, lr=lr)
    if name=='rmsprop': return optim.RMSprop(params, lr=lr, momentum=0.9)
    raise ValueError(name)

# ---------- Train/Eval ----------
def train_eval(hp, epochs=2):
    train_loader, test_loader = load_mnist(hp['batch'])
    model = FNN(hidden=hp['hidden'], dropout=hp['dropout']).to(DEVICE)
    crit = nn.CrossEntropyLoss()
    opt = get_optimizer(hp['opt'], model.parameters(), hp['lr'])
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); out = model(xb); loss = crit(out, yb); loss.backward(); opt.step()
    # eval
    model.eval(); correct=0; total=0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            pred = out.argmax(1)
            correct += (pred==yb).sum().item()
            total += yb.size(0)
    acc = correct/total
    return acc

# ---------- Fitness ----------
def fitness(ind):
    hp = decode(ind)
    acc = train_eval(hp, epochs=1)  # 1-epoch for speed; increase later
    return (acc,)

# ---------- GA ----------
def run_ga(pop=8, gens=3):
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register('attr_float', random.random)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_float, 5)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', fitness)
    toolbox.register('mate', tools.cxUniform, indpb=0.5)
    toolbox.register('mutate', tools.mutGaussian, mu=0.0, sigma=0.1, indpb=0.5)
    toolbox.register('select', tools.selTournament, tournsize=3)
    pop = toolbox.population(n=pop)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.3, ngen=gens, halloffame=hof, verbose=True)
    return hof[0].fitness.values[0], decode(hof[0])

# ---------- DE ----------
def clamp01(x): return max(0.0, min(1.0, x))
def run_de(pop=8, gens=3, F=0.5, CR=0.9):
    P = [np.random.rand(5) for _ in range(pop)]
    fit = [fitness(p.tolist())[0] for p in P]
    for g in range(gens):
        for i in range(pop):
            a,b,c = np.random.choice([j for j in range(pop) if j!=i], 3, replace=False)
            mutant = P[a] + F*(P[b]-P[c])
            trial = np.array([mutant[j] if random.random()<CR else P[i][j] for j in range(5)])
            trial = np.array([clamp01(v) for v in trial])
            f_trial = fitness(trial.tolist())[0]
            if f_trial > fit[i]:
                P[i], fit[i] = trial, f_trial
    bi = int(np.argmax(fit))
    return fit[bi], decode(P[bi].tolist())

# ---------- PSO ----------
def run_pso(n=8, iters=3, w=0.7, c1=1.5, c2=1.5):
    dim=5
    X = np.random.rand(n,dim); V = np.zeros((n,dim))
    pbest = X.copy(); pfit = np.array([fitness(x.tolist())[0] for x in X])
    gi = int(np.argmax(pfit)); gbest = pbest[gi].copy(); gfit = pfit[gi]
    for t in range(iters):
        for i in range(n):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            V[i] = w*V[i] + c1*r1*(pbest[i]-X[i]) + c2*r2*(gbest-X[i])
            X[i] = np.clip(X[i]+V[i], 0.0, 1.0)
            f = fitness(X[i].tolist())[0]
            if f > pfit[i]:
                pfit[i], pbest[i] = f, X[i].copy()
                if f > gfit:
                    gfit, gbest = f, X[i].copy()
    return gfit, decode(gbest.tolist())

# ---------- Main ----------
def main():
    results = []
    print("Running GA..."); acc_ga, hp_ga = run_ga(pop=6, gens=2)
    results.append(("GA", acc_ga, hp_ga))
    print("Running DE..."); acc_de, hp_de = run_de(pop=6, gens=2)
    results.append(("DE", acc_de, hp_de))
    print("Running PSO..."); acc_pso, hp_pso = run_pso(n=6, iters=2)
    results.append(("PSO", acc_pso, hp_pso))

    with open('results.csv','w', newline='') as f:
        w = csv.writer(f); w.writerow(["algo","best_acc","best_params_json"])
        for algo, acc, hp in results:
            w.writerow([algo, f"{acc:.4f}", json.dumps(hp)])

    print("Done. Results in results.csv.")
    for r in results:
        print(r)

if __name__ == "__main__":
    main()
