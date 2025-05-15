import kgbench as kg
import torch
import torch.nn.functional as F

from experiments.models import LGCN


def go(model_name, name, lr, wd, l2, epochs, prune, optimizer, final, emb_dim, weights_size=None, rp=None, ldepth=None,
       lwidth=None, bases=None,
       printnorms=None, trial=None, wandb=None):
    # Manually created sample graph
    triples = torch.tensor([[1, 0, 0],
                            [2, 1, 0],
                            [2, 2, 1],
                            [3, 0, 0],
                            [4, 2, 2],
                            [5, 1, 2],
                            [5, 0, 4],
                            [5, 1, 4]])
    num_entities = 6
    num_relations = 3
    num_classes = 2
    training = torch.tensor([[5, 1],
                             [1, 0],
                             [0, 0],
                             [4, 1],
                             [5, 1]])
    withheld = torch.tensor([[3, 1],
                             [2, 0]])

    print(f'Model: {model_name}, Dataset: {name}, ')
    print(
        f'Parameters: lr={lr}, wd={wd}, l2={l2}, epochs={epochs}, prune={prune}, optimizer={optimizer}, final={final},'
        f' emb_dim={emb_dim}, weights_size={weights_size}, rp={rp}, ldepth={ldepth}, lwidth={lwidth}, bases={bases}')

    kg.tic()

    # Initialize LGCN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda"):
        print('Using CUDA')

    model = LGCN(triples, num_nodes=num_entities, num_rels=num_relations,
                 num_classes=num_classes,
                 emb_dim=emb_dim, rp=rp, ldepth=ldepth, lwidth=lwidth, bases=bases).to(device)

    print(f'Model created in {kg.toc():.3}s')

    # Move data to the same device as the model
    training = training.to(device)
    withheld = withheld.to(device)

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
        idxt, clst = training[:, 0], training[:, 1].to(torch.int64)
        idxw, clsw = withheld[:, 0], withheld[:, 1].to(torch.int64)

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

        # Print epoch statistics
        print(
            f'Epoch {e:02}: \t\t loss {loss:.4f}, \t\t train acc {training_acc:.2f}, \t\t withheld acc'
            f' {withheld_acc:.2f}, '
            f'\t\t ({kg.toc():.3}s)')

    print(f'\nTraining complete! (total time: {kg.toc() / 60:.2f}m)')

    return withheld_acc


if __name__ == '__main__':
    # LGCN
    go(model_name='lgcn', name='sample', lr=0.001, wd=0.0, l2=0.0, epochs=200, prune=True, optimizer='adam',
       final=False, emb_dim=128, weights_size=None, rp=16, ldepth=0, lwidth=128, bases=None, printnorms=None)
