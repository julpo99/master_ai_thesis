from collections import Counter

import kgbench as kg
import torch
import torch.nn.functional as F

from models import RGCN, LGCN, LGCN2


def go(model_name='rgcn', name='amplus', lr=0.01, wd=0.0, l2=0.0, epochs=50, prune=False, optimizer='adam',
       final=False, emb_dim=16, weights_size=16, rp=16, ldepth=0, lwidth=64, bases=None, printnorms=None):
    # Load dataset
    data = kg.load(name, torch=True, prune_dist=2 if prune else None, final=final)

    print(f'Model: {model_name}')
    print(f'Loaded {data.triples.size(0)} triples, {data.num_entities} entities, {data.num_relations} relations')

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
                     emb_dim=emb_dim, weights_size=weights_size).to(device)
    elif model_name == 'lgcn2':
        model = LGCN2(data.triples, num_nodes=data.num_entities, num_rels=data.num_relations,
                     num_classes=data.num_classes,
                     emb_dim=emb_dim, weights_size=weights_size, rp=rp, ldepth=ldepth, lwidth=lwidth).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    print(f'Model created in {kg.toc():.3}s')

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


if __name__ == '__main__':
    model_to_run = 'lgcn'  # or 'lgcn'

    if model_to_run == 'rgcn':
        # RGCN
        go(model_name='rgcn', name='amplus', lr=0.01, wd=0.0, l2=0.0005, epochs=50, prune=True, optimizer='adam',
           final=False, emb_dim=10, weights_size=16, bases=40, printnorms=None)
    elif model_to_run == 'lgcn':
        # LGCN
        go(model_name='lgcn', name='amplus', lr=0.01, wd=0.0, l2=0.0, epochs=50, prune=True, optimizer='adam',
           final=False, emb_dim=1600, weights_size=16, bases=None, printnorms=None)
    elif model_to_run == 'lgcn2':
        # LGCN2
        go(model_name='lgcn2', name='amplus', lr=0.01, wd=0.0, l2=0.0005, epochs=150, prune=True, optimizer='adam',
           final=False, emb_dim=16, weights_size=0, rp=16, ldepth=0, lwidth=64, bases=None, printnorms=None)
    else:
        raise ValueError(f"Unknown model name: {model_to_run}")
