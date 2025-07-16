import kgbench as kg
import torch
import torch.nn.functional as F

from configs.default_config import DEFAULT_CONFIG
from experiments.model_builder import build_model

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

    # Manually created sample graph

    # Small sample graph
    triples = torch.tensor([[0, 0, 1],
                            [0, 1, 2],
                            [1, 1, 2],
                            [2, 2, 3]])
    num_entities = 4
    num_relations = 3
    num_classes = 2
    training = torch.tensor([[0, 1],
                             [1, 0],
                             [2, 1]])
    withheld = torch.tensor([[3, 1]])

    # Big sample graph
    # triples = torch.tensor([[1, 0, 0],
    #                         [2, 1, 0],
    #                         [2, 2, 1],
    #                         [3, 0, 0],
    #                         [4, 2, 2],
    #                         [5, 1, 2],
    #                         [5, 0, 4],
    #                         [5, 1, 4]])
    # num_entities = 6
    # num_relations = 3
    # num_classes = 2
    # training = torch.tensor([[5, 1],
    #                          [1, 0],
    #                          [0, 0],
    #                          [4, 1]])
    # withheld = torch.tensor([[3, 1],
    #                          [2, 0]])

    print(f"\nModel: {config['model_name']}, Dateset: sample graph")
    print(
        f"Parameters: lr={config['lr']}, wd={config['wd']}, l2={config['l2']}, epochs={config['epochs']}, "
        f"optimizer={config['optimizer']}, emb_dim={config['emb_dim']}, weights_size={config.get('weights_size')}, "
        f"rp={config.get('rp')}, ldepth={config.get('ldepth')}, lwidth={config.get('lwidth')}, "
        f"bases={config.get('bases')}, dropout={config.get('dropout')}, enrich_flag={config['enrich_flag']}"
    )
    print(f'\nLoaded {triples.size(0)} triples, {num_entities} entities, {num_relations} relations\n')

    # Initialize model
    kg.tic()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cuda'):
        print('Using CUDA')


    model = build_model(config, triples, num_entities, num_relations, num_classes).to(device)

    print(f"Model {config['model_name']} created in {kg.toc():.3}s\n")

    # Move data to the same device as the model
    training = training.to(device)
    withheld = withheld.to(device)

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
        idxt, clst = training[:, 0], training[:, 1].long()
        idxw, clsw = withheld[:, 0], withheld[:, 1].long()

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


if __name__ == '__main__':
    # SETUP: only modify this line:
    model_to_run = 'rgcn'  # options: "rgcn", "lgcn_rel_emb"

    train(DEFAULT_CONFIG[model_to_run])