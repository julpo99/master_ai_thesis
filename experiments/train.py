import wandb
import kgbench as kg
import optuna
import torch
import torch.nn.functional as F

from configs.default_config import DEFAULT_CONFIG
from configs.best_config import BEST_CONFIG
from configs.optuna_config import OPTUNA_SEARCH_SPACE
from configs.experiment_config import EXPERIMENT_CONFIG
from model_builder import build_model

class EarlyStopping:
    def __init__(self, patience=20, mode='max', min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, current):
        if self.best is None:
            self.best = current
            return False

        improvement = (current - self.best) if self.mode == 'max' else (self.best - current)
        if improvement > self.min_delta:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True
            return True

        return False


def train(config, trial=None, wandb=None):
    # Load dataset
    dataset_config = EXPERIMENT_CONFIG['dataset']
    data = kg.load(
        dataset_config['name'],
        torch=True,
        prune_dist=2 if dataset_config['prune'] else None,
        final=dataset_config['final'])

    print(f"\nModel: {config['model_name']}, Dateset: {dataset_config['name']}")

    print(
        f"Parameters: lr={config['lr']}, wd={config['wd']}, l2={config['l2']}, epochs={config['epochs']}, "
        f"optimizer={config['optimizer']}, emb_dim={config['emb_dim']}, weights_size={config.get('weights_size')}, "
        f"rp={config.get('rp')}, ldepth={config.get('ldepth')}, lwidth={config.get('lwidth')}, "
        f"bases={config.get('bases')}, dropout={config.get('dropout')}, enrich_flag={config['enrich_flag']}"
    )
    print(f'\nLoaded {data.triples.size(0)} triples, {data.num_entities} entities, {data.num_relations} relations\n')

    # Initialize model
    kg.tic()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cuda'):
        print('Using CUDA')

    model = build_model(config, data.triples, data.num_entities, data.num_relations, data.num_classes).to(device)

    print(f"Model {config['model_name']} created in {kg.toc():.3}s\n")

    # Move data to the same device as the model
    data.training = data.training.to(device)
    data.withheld = data.withheld.to(device)

    # Select optimizer
    optimizer_class = torch.optim.Adam if config['optimizer'] == 'adam' else torch.optim.AdamW
    optimizer = optimizer_class(model.parameters(), lr=config['lr'], weight_decay=config['wd'])

    kg.tic()

    early_stopper = EarlyStopping(patience=30, mode='max', min_delta=0.001)
    best_withh_acc = 0.0
    best_epoch = -1

    for e in range(config['epochs']):


        kg.tic()

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model()

        # Extract indices for training & withheld sets
        idxt, clst = data.training[:, 0], data.training[:, 1].long()
        idxw, clsw = data.withheld[:, 0], data.withheld[:, 1].long()

        # Compute loss
        out_train = output[idxt, :]
        loss = F.cross_entropy(out_train, clst)

        if (l2 := config['l2']) > 0.0:
            loss += l2 * model.penalty()

        # Backward pass (compute gradients)
        loss.backward()

        # Update weights
        optimizer.step()

        # Compute performance metrics
        with torch.no_grad():
            train_acc = (output[idxt].argmax(1) == clst).float().mean().item()
            withh_acc = (output[idxw].argmax(1) == clsw).float().mean().item()

            if withh_acc > best_withh_acc:
                best_withh_acc = withh_acc
                best_epoch = e


            if trial is not None:
                trial.report(withh_acc, e)

            if wandb is not None:
                # Log metrics to wandb
                wandb.log(data={'loss': loss.item(), 'train_acc': train_acc, 'withheld_acc': withh_acc}, step=e)

        # Print epoch statistics
        print(
            f'Epoch {e:03}: loss = {loss:.2f}, train_acc = {train_acc:.2f}, withheld_acc = {withh_acc:.2f}, ({kg.toc():.2}s)')

        if early_stopper(withh_acc):
            print(f"Early stopping at epoch {e} (best withheld_acc = {early_stopper.best:.4f})")
            break

    print(f'\nTraining complete! (total time: {kg.toc() / 60:.2f}m)')
    print(f"Best withheld_acc = {best_withh_acc:.4f} @ epoch {best_epoch}")

    if wandb is not None:
        wandb.run.summary['final accuracy'] = withh_acc
        wandb.run.summary['state'] = 'completed'
        wandb.finish(quiet=True)

    return best_withh_acc


def optuna_objective(trial, model_name):
    search_space = OPTUNA_SEARCH_SPACE[model_name]
    params = {}

    for param_name, search in search_space.items():
        if search[0] == 'float':
            log_flag = search[3] == 'log' if len(search) > 3 else False
            params[param_name] = trial.suggest_float(param_name, search[1], search[2], log=log_flag)
        elif search[0] == 'int':
            params[param_name] = trial.suggest_int(param_name, search[1], search[2])
        elif search[0] == 'categorical':
            params[param_name] = trial.suggest_categorical(param_name, search[1])
        else:
            raise ValueError(f"Unknown search type for param {param_name}")

    params['trial.number'] = trial.number
    params['model_name'] = model_name

    dataset_cfg = EXPERIMENT_CONFIG['dataset']
    params['name'] = dataset_cfg['name']
    params['prune'] = dataset_cfg['prune']
    params['final'] = dataset_cfg['final']

    wandb.init(
        project=f"{model_name}_optuna",
        entity="julpo99-vrije-universiteit-amsterdam",
        config=params,
        reinit="default"
    )

    # Launch training
    withh_acc = train(params, trial=trial, wandb=wandb)

    return withh_acc


if __name__ == '__main__':
    # SETUP: only modify these two lines:
    model_to_run = 'rgcn' # options: "rgcn", "rgnn_emb", "lgcn", "lgcn_rel_emb"
    mode = 'best'  # options: "default", "best", "optuna"

    if mode == 'default':
        config = DEFAULT_CONFIG[model_to_run]
        train(config)

    elif mode == 'best':
        config = BEST_CONFIG[model_to_run]
        train(config)

    elif mode == 'optuna':
        study = optuna.create_study(
            direction='maximize',
            study_name=f"{model_to_run}_optuna"
        )
        study.optimize(lambda trial: optuna_objective(trial, model_to_run))

    else:
        raise ValueError(f"Unknown mode '{mode}'. Please choose from: default, best, optuna.")
