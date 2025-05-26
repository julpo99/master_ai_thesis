from collections import Counter

import wandb
import kgbench as kg
import optuna
import torch
import torch.nn.functional as F

from models import RGCN, RGCN_EMB, LGCN, LGCN_REL_EMB


def go(model_name, name, lr, wd, l2, epochs, prune, optimizer, final, emb_dim, weights_size=None, rp=None, ldepth=None,
       lwidth=None, bases=None,
       printnorms=None, trial=None, wandb=None):
    # Load dataset
    data = kg.load(name, torch=True, prune_dist=2 if prune else None, final=final)

    print(f'\nModel: {model_name}, Dateset: {name}')
    print(
        f'Parameters: lr={lr}, wd={wd}, l2={l2}, epochs={epochs}, prune={prune}, optimizer={optimizer}, final={final},'
        f' emb_dim={emb_dim}, weights_size={weights_size}, rp={rp}, ldepth={ldepth}, lwidth={lwidth}, bases={bases}')

    print(f'\nLoaded {data.triples.size(0)} triples, {data.num_entities} entities, {data.num_relations} relations\n')

    kg.tic()

    # Initialize R-GCN model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cuda'):
        print('Using CUDA')

    if model_name == 'rgcn':
        model = RGCN(data.triples, num_nodes=data.num_entities, num_rels=data.num_relations,
                     num_classes=data.num_classes,
                     emb_dim=emb_dim, bases=bases).to(device)
    elif model_name == 'rgcn_emb':
        model = RGCN_EMB(data.triples, num_nodes=data.num_entities, num_rels=data.num_relations,
                     num_classes=data.num_classes,
                     emb_dim=emb_dim, weights_size=weights_size, bases=bases).to(device)
    elif model_name == 'lgcn':
        model = LGCN(data.triples, num_nodes=data.num_entities, num_rels=data.num_relations,
                      num_classes=data.num_classes,
                      emb_dim=emb_dim, rp=rp, ldepth=ldepth, lwidth=lwidth).to(device)
    elif model_name == 'lgcn_rel_emb':
        model = LGCN_REL_EMB(data.triples, num_nodes=data.num_entities, num_rels=data.num_relations,
                             num_classes=data.num_classes,
                             emb_dim=emb_dim, rp=rp).to(device)
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    print(f'Model {model_name} created in {kg.toc():.3}s\n')

    # Move data to the same device as the model
    data.training = data.training.to(device)
    data.withheld = data.withheld.to(device)

    # Select optimizer
    optimizers = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW
    }
    if optimizer not in optimizers:
        raise ValueError(f'Unknown optimizer: {optimizer}')

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

        # Backward pass (compute gradients)
        loss.backward()

        # Update weights
        opt.step()

        # Compute performance metrics
        with torch.no_grad():
            preds_train = out[idxt].argmax(dim=1)
            preds_withheld = out[idxw].argmax(dim=1)

            training_acc = torch.sum(preds_train == clst).item() / idxt.size(0)
            withheld_acc = torch.sum(preds_withheld == clsw).item() / idxw.size(0)

            if trial is not None:
                trial.report(withheld_acc, e)

                # Log metrics to wandb
                wandb.log(data={'loss': loss.item(), 'train_acc': training_acc, 'withheld_acc': withheld_acc}, step=e)


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

    if wandb is not None:
        wandb.run.summary['final accuracy'] = withheld_acc
        wandb.run.summary['state'] = 'completed'
        wandb.finish(quiet=True)

    return withheld_acc


def objective_rgcn(trial):
    lr = trial.suggest_float('lr', 0.01, 0.1, log=True)
    wd = trial.suggest_float('wd', 0.0001, 0.03, log=True)
    l2 = trial.suggest_float('l2', 0.00001, 0.001, log=True)
    epochs = trial.suggest_int('epochs', 30, 200)
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'adamw'])
    emb_dim = trial.suggest_int('emb_dim', 8, 64)
    bases = trial.suggest_categorical('bases', [None] + list(range(1, 51)))

    config = dict(trial.params)
    config['trial.number'] = trial.number
    wandb.init(project='rgcn_optuna', entity='julpo99-vrije-universiteit-amsterdam', config=config, reinit='default')

    withheld_acc = go(model_name='rgcn', name='amplus', lr=lr, wd=wd, l2=l2, epochs=epochs, prune=True,
                      optimizer=optimizer, final=False, emb_dim=emb_dim, bases=bases, printnorms=None, trial=trial, wandb=wandb)

    return withheld_acc


def objective_rgcn_emb(trial):
    lr = trial.suggest_float('lr', 0.001, 0.1, log=True)
    wd = trial.suggest_float('wd', 0.00001, 0.01, log=True)
    l2 = trial.suggest_float('l2', 0.00001, 0.001, log=True)
    epochs = trial.suggest_int('epochs', 5, 60)
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'adamw'])
    emb_dim = trial.suggest_int('emb_dim', 512, 2048)
    weights_size = trial.suggest_int('weights_size', 32, 128)
    bases = trial.suggest_categorical('bases', [None] + list(range(1, 65)))

    config = dict(trial.params)
    config['trial.number'] = trial.number
    wandb.init(project='rgcn_emb_optuna', entity='julpo99-vrije-universiteit-amsterdam', config=config, reinit='default')

    withheld_acc = go(model_name='rgcn_emb', name='amplus', lr=lr, wd=wd, l2=l2, epochs=epochs, prune=True,
                      optimizer=optimizer, final=False, emb_dim=emb_dim, weights_size=weights_size, bases=bases,
                      printnorms=None, trial=trial, wandb=wandb)

    return withheld_acc


def objective_lgcn(trial):
    lr = trial.suggest_float('lr', 0.01, 0.1, log=True)
    wd = trial.suggest_float('wd', 0.00001, 0.0001, log=True)
    l2 = trial.suggest_float('l2', 0.000001, 0.0001, log=True)
    epochs = trial.suggest_int('epochs', 60, 150)
    # optimizer = trial.suggest_categorical('optimizer', ['adam', 'adamw'])
    optimizer = 'adam'
    emb_dim = trial.suggest_int('emb_dim', 64, 128)
    rp = trial.suggest_int('rp', 1, 16)
    # ldepth = trial.suggest_int('ldepth', 1, 4)
    ldepth = 1
    lwidth = trial.suggest_int('lwidth', 128, 256)
    bases = trial.suggest_categorical('bases', [None] + list(range(1, 51)))

    config = dict(trial.params)
    config['trial.number'] = trial.number
    wandb.init(project='lgcn_optuna', entity='julpo99-vrije-universiteit-amsterdam', config=config, reinit='default')

    withheld_acc = go(model_name='lgcn', name='amplus', lr=lr, wd=wd, l2=l2, epochs=epochs, prune=True,
                      optimizer=optimizer, final=False, emb_dim=emb_dim, rp=rp, ldepth=ldepth, lwidth=lwidth,
                      bases=bases,
                      printnorms=None, trial=trial, wandb=wandb)

    return withheld_acc


if __name__ == '__main__':
    model_to_run = 'lgcn_optuna'
    # 'rgcn', 'rgcn_best', 'rgcn_emb', 'rgcn_emb_best', 'lgcn', 'lgcn_best'

    # 'rgcn_optuna', 'rgcn_emb_optuna', 'lgcn_optuna'

    # 'lgcn_rel_emb', 'lgcn_rel_emb_best'

    if model_to_run == 'rgcn':
        # RGCN
        go(model_name='rgcn', name='amplus', lr=0.01, wd=0.0, l2=0.0005, epochs=50, prune=True, optimizer='adam',
           final=False, emb_dim=16, bases=40, printnorms=None)

    elif model_to_run == 'rgcn_best':
        # RGCN Best (optuna)
        # go(model_name='rgcn', name='amplus', lr=0.03496818515661486, wd=0.000384113466755141, l2=0.0009027947821005017,
        #    epochs=50, prune=True, optimizer='adam',
        #    final=False, emb_dim=29, bases=15, printnorms=None)
        go(model_name='rgcn', name='amplus', lr=0.05499010733725328, wd=0.00022666309964608877, l2=0.000514707867508644,
           epochs=163, prune=True, optimizer='adamw', final=False, emb_dim=31, bases=None, printnorms=None)


    elif model_to_run == 'rgcn_emb':
        # RGCN_EMB
        go(model_name='rgcn_emb', name='amplus', lr=0.01, wd=0.0, l2=0.0005, epochs=50, prune=True, optimizer='adam',
           final=False, emb_dim=1600, weights_size=16, bases=None, printnorms=None, wandb=None)

    elif model_to_run == 'rgcn_emb_best':
        # RGCN_EMB Best (optuna)
        go(model_name='rgcn_emb', name='amplus', lr=0.0016366409775459736, wd=0.0005755564025828806, l2=4.854651125478105e-05, epochs=45,
           prune=True, optimizer='adam',
           final=False, emb_dim=1236, weights_size=49, bases=29, printnorms=None)

    elif model_to_run == 'lgcn':
        # LGCN
        go(model_name='lgcn', name='amplus', lr=0.001, wd=0.0, l2=0.0, epochs=200, prune=True, optimizer='adam',
           final=False, emb_dim=128, weights_size=None, rp=16, ldepth=1, lwidth=128, bases=None, printnorms=None)
    elif model_to_run == 'lgcn_best':
        # LGCN Best (optuna)
        go(model_name='lgcn', name='amplus', lr=0.07853833444430745, wd=1.964167180340962e-05, l2=1.3917838028734193e-05,
              epochs=90, prune=True, optimizer='adam',
              final=False, emb_dim=80, weights_size=None, rp=12, ldepth=1, lwidth=125, bases=None, printnorms=None)


    elif model_to_run == 'lgcn_rel_emb':
        # LGCN with relation embeddings
        go(model_name='lgcn_rel_emb', name='amplus', lr=0.001, wd=0.0, l2=0.0, epochs=200, prune=True, optimizer='adam',
           final=False, emb_dim=128, weights_size=None, rp=16, ldepth=0, lwidth=128, bases=None,
           printnorms=None)

    elif model_to_run == 'rgcn_optuna':
        # Optuna rgcn
        study = optuna.create_study(
            direction='maximize',
            study_name='rgcn_optuna',
        )

        study.optimize(objective_rgcn, n_trials=10000)
        print('Best trial:')
        trial = study.best_trial
        print(trial)

    elif model_to_run == 'rgcn_emb_optuna':
        # Optuna rgcn_emb
        study = optuna.create_study(
            direction='maximize',
            study_name='rgcn_emb_optuna',
        )
        study.optimize(objective_rgcn_emb, n_trials=10000)
        print('Best trial:')
        trial = study.best_trial
        print(trial)

    elif model_to_run == 'lgcn_optuna':
        # Optuna lgcn
        study = optuna.create_study(
            direction='maximize',
            study_name='lgcn_optuna',
        )
        study.optimize(objective_lgcn, n_trials=10000)
        print('Best trial:')
        trial = study.best_trial
        print(trial)
    else:
        raise ValueError(f'Unknown model name: {model_to_run}')
