from collections import Counter

import kgbench as kg
import optuna
import torch
import torch.nn.functional as F

from models import RGCN, LGCN, LGCN2


def go(model_name, name, lr, wd, l2, epochs, prune, optimizer, final, emb_dim, weights_size=None, rp=None, ldepth=None,
       lwidth=None, bases=None,
       printnorms=None):
    # Load dataset
    data = kg.load(name, torch=True, prune_dist=2 if prune else None, final=final)

    print(f'\nModel: {model_name}, Dateset: {name}')
    print(
        f'Parameters: lr={lr}, wd={wd}, l2={l2}, epochs={epochs}, prune={prune}, optimizer={optimizer}, final={final},'
        f' emb_dim={emb_dim}, weights_size={weights_size}, rp={rp}, ldepth={ldepth}, lwidth={lwidth}, bases={bases}')

    print(f'\nLoaded {data.triples.size(0)} triples, {data.num_entities} entities, {data.num_relations} relations\n')

    kg.tic()

    # Initialize R-GCN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda"):
        print('Using CUDA')

    if model_name == 'rgcn':
        model = RGCN(data.triples, num_nodes=data.num_entities, num_rels=data.num_relations,
                     num_classes=data.num_classes,
                     emb_dim=emb_dim, bases=bases).to(device)
    elif model_name == 'lgcn':
        model = LGCN(data.triples, num_nodes=data.num_entities, num_rels=data.num_relations,
                     num_classes=data.num_classes,
                     emb_dim=emb_dim, weights_size=weights_size, bases=bases).to(device)
    elif model_name == 'lgcn2':
        model = LGCN2(data.triples, num_nodes=data.num_entities, num_rels=data.num_relations,
                      num_classes=data.num_classes,
                      emb_dim=emb_dim, rp=rp, ldepth=ldepth, lwidth=lwidth, bases=bases).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    print(f'Model {model_name} created in {kg.toc():.3}s\n')

    # Move data to the same device as the model
    data.training = data.training.to(device)
    data.withheld = data.withheld.to(device)

    # Select optimizer
    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW
    }
    if optimizer not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    opt = optimizers[optimizer](model.parameters(), lr=lr, weight_decay=wd)

    kg.tic()

    for e in range(epochs):
        kg.tic()

        # Zero gradients
        opt.zero_grad()

        # Forward pass
        out = model()

        # Extract indices for training & withheld sets
        idxt, clst = data.training[:, 0], data.training[:, 1].to(torch.int64)
        idxw, clsw = data.withheld[:, 0], data.withheld[:, 1].to(torch.int64)

        # Compute loss
        out_train = out[idxt, :]
        loss = F.cross_entropy(out_train, clst, reduction='mean')

        if l2 > 0.0:
            loss += l2 * model.penalty()

        # Backward pass (compute  gradients)
        loss.backward()

        # Update weights
        opt.step()

        # Compute performance metrics
        with torch.no_grad():
            preds_train = out[idxt].argmax(dim=1)
            preds_withheld = out[idxw].argmax(dim=1)

            training_acc = torch.sum(preds_train == clst).item() / idxt.size(0)
            withheld_acc = torch.sum(preds_withheld == clsw).item() / idxw.size(0)

        # Print epoch statistics
        print(
            f'Epoch {e:02}: \t\t loss {loss:.4f}, \t\t train acc {training_acc:.2f}, \t\t withheld acc'
            f' {withheld_acc:.2f}, '
            f'\t\t ({kg.toc():.3}s)')

        # Print relation norms if requested
        if printnorms is not None:
            nr = data.num_relations

            def print_norms(weights, layer_num):
                ctr = Counter()

                for r in range(nr):
                    ctr[data.i2r[r]] = weights[r].norm().item()
                    ctr['inv_' + data.i2r[r]] = weights[r + nr].norm().item()  # Handle inverse relations

                print(f'Relations with largest weight norms in layer {layer_num}.')
                for rel, w in ctr.most_common(printnorms):
                    print(f'    norm {w:.4f} for {rel}')

            print_norms(model.weights1, 1)
            print_norms(model.weights2, 2)

    print(f'\nTraining complete! (total time: {kg.toc() / 60:.2f}m)')

    return withheld_acc


def objective_rgcn(trial):
    lr = trial.suggest_float('lr', 0.01, 0.1, log=True)
    wd = trial.suggest_float('wd', 0.0001, 0.1, log=True)
    l2 = trial.suggest_float('l2', 0.00001, 0.001, log=True)
    epochs = trial.suggest_int('epochs', 30, 200)
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'adamw'])
    emb_dim = trial.suggest_int('emb_dim', 8, 32)
    bases = trial.suggest_categorical('bases', [None] + list(range(1, 51)))

    withheld_acc = go(model_name='rgcn', name='amplus', lr=lr, wd=wd, l2=l2, epochs=epochs, prune=True,
                      optimizer=optimizer, final=False, emb_dim=emb_dim, bases=bases, printnorms=None)

    return withheld_acc


def objective_lgcn(trial):
    lr = trial.suggest_float('lr', 0.001, 0.1, log=True)
    wd = trial.suggest_float('wd', 0.00001, 0.1, log=True)
    l2 = trial.suggest_float('l2', 0.00001, 0.001, log=True)
    epochs = trial.suggest_int('epochs', 30, 200)
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'adamw'])
    emb_dim = trial.suggest_int('emb_dim', 512, 2048)
    weights_size = trial.suggest_int('weights_size', 1, 64)
    bases = trial.suggest_categorical('bases', [None] + list(range(1, 2)))

    withheld_acc = go(model_name='lgcn', name='amplus', lr=lr, wd=wd, l2=l2, epochs=epochs, prune=True,
                      optimizer=optimizer, final=False, emb_dim=emb_dim, weights_size=weights_size, bases=bases,
                      printnorms=None)

    return withheld_acc


def objective_lgcn2(trial):
    lr = trial.suggest_float('lr', 0.001, 0.1, log=True)
    wd = trial.suggest_float('wd', 0.00001, 0.1, log=True)
    l2 = trial.suggest_float('l2', 0.00001, 0.001, log=True)
    epochs = trial.suggest_int('epochs', 30, 200)
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'adamw'])
    emb_dim = trial.suggest_int('emb_dim', 128, 1024)
    rp = trial.suggest_int('rp', 1, 32)
    ldepth = trial.suggest_int('ldepth', 1, 16)
    lwidth = trial.suggest_int('lwidth', 16, 256)
    bases = trial.suggest_categorical('bases', [None] + list(range(1, 51)))

    withheld_acc = go(model_name='lgcn2', name='amplus', lr=lr, wd=wd, l2=l2, epochs=epochs, prune=True,
                      optimizer=optimizer, final=False, emb_dim=emb_dim, rp=rp, ldepth=ldepth, lwidth=lwidth,
                      bases=bases,
                      printnorms=None)

    return withheld_acc


if __name__ == '__main__':
    model_to_run = 'lgcn_optuna'  # 'rgcn', 'lgcn', 'lgcn_best', 'lgcn2', 'rgcn_optuna', 'lgcn_optuna'

    if model_to_run == 'rgcn':
        # RGCN
        go(model_name='rgcn', name='amplus', lr=0.01, wd=0.0, l2=0.0005, epochs=50, prune=True, optimizer='adam',
           final=False, emb_dim=16, bases=40, printnorms=None)

    elif model_to_run == 'rgcn_best':
        # RGCN Best (optuna)
        go(model_name='rgcn', name='amplus', lr=0.03496818515661486, wd=0.000384113466755141, l2=0.0009027947821005017,
           epochs=50, prune=True, optimizer='adam',
           final=False, emb_dim=29, bases=15, printnorms=None)



    elif model_to_run == 'lgcn':
        # LGCN
        go(model_name='lgcn', name='amplus', lr=0.01, wd=0.0, l2=0.0005, epochs=50, prune=True, optimizer='adam',
           final=False, emb_dim=1600, weights_size=16, bases=20, printnorms=None)

    elif model_to_run == 'lgcn_best':
        # LGCN Best (optuna)
        go(model_name='lgcn', name='amplus', lr=0.01, wd=0.010123139133733597, l2=1.2733338809274765e-06, epochs=50,
           prune=True, optimizer='adam',
           final=False, emb_dim=693, weights_size=16, bases=34, printnorms=None)



    elif model_to_run == 'lgcn2':
        # LGCN2
        go(model_name='lgcn2', name='amplus', lr=0.001, wd=0.0, l2=0.0, epochs=200, prune=True, optimizer='adam',
           final=False, emb_dim=128, weights_size=0, rp=16, ldepth=8, lwidth=128, bases=None, printnorms=None)




    elif model_to_run == 'rgcn_optuna':
        # Optuna rgcn
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_rgcn, n_trials=10000)
        print('Best trial:')
        trial = study.best_trial
        print(trial)

    elif model_to_run == 'lgcn_optuna':
        # Optuna rgcn
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_lgcn, n_trials=10000)
        print('Best trial:')
        trial = study.best_trial
        print(trial)

    elif model_to_run == 'lgcn2_optuna':
        # Optuna lgcn2
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_lgcn2, n_trials=10000)
        print('Best trial:')
        trial = study.best_trial
        print(trial)
    else:
        raise ValueError(f"Unknown model name: {model_to_run}")
