from collections import Counter

import kgbench as kg
import torch
import torch.nn.functional as F

from experiments.models import RGCN, LGCN, LGCN2


def go(model_name='rgcn', name='amplus', lr=0.01, wd=0.0, l2=0.0, epochs=50, prune=False, optimizer='adam',
       final=False, emb_dim=16, weights_size=16, rp=16, ldepth=0, lwidth=64, bases=None, printnorms=None):
    # Manually created sample graph
    triples = torch.tensor([[1, 0, 0],
                            [2, 1, 0],
                            [2, 2, 1],
                            [3, 0, 0],
                            [4, 2, 2],
                            [5, 1, 2],
                            [5, 0, 4]])
    num_entities = 6
    num_relations = 3
    num_classes = 2
    training = torch.tensor([[5, 1],
                             [1, 0],
                             [0, 0],
                             [4, 1]])
    withheld = torch.tensor([[3, 1],
                             [2, 0]])

    print(f'Model: {model_name}')
    print(f'Loaded {triples.size(0)} triples, {num_entities} entities, {num_relations} relations')

    kg.tic()

    # Initialize R-GCN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda"):
        print('Using CUDA')

    model = LGCN(triples, num_nodes=num_entities, num_rels=num_relations,
                 num_classes=num_classes,
                 emb_dim=emb_dim, weights_size=weights_size, bases=bases).to(device)

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

    print(f'\nTraining complete! (total time: {kg.toc() / 60:.2f}m)')

    return withheld_acc


if __name__ == '__main__':
    # LGCN
    go(model_name='lgcn', name='amplus', lr=0.01, wd=0.0, l2=0.0005, epochs=150, prune=True, optimizer='adam',
       final=False, emb_dim=1600, weights_size=16, bases=20, printnorms=None)
