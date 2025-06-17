import torch
from models import RGCN, RGCN_EMB, LGCN, LGCN_REL_EMB


def build_model(config: dict, triples: torch.Tensor, num_nodes: int, num_rels: int, num_classes: int):
    """
    Build the appropriate model given the config dictionary.

    Args:
        config (dict): configuration dictionary
        triples (torch.Tensor): dataset triples
        num_nodes (int): number of nodes
        num_rels (int): number of relations
        num_classes (int): number of classes

    Returns:
        model (nn.Module): the initialized model
    """

    model_name = config['model_name']

    if model_name == 'rgcn':
        model = RGCN(
            triples=triples,
            num_nodes=num_nodes,
            num_rels=num_rels,
            num_classes=num_classes,
            emb_dim=config.get('emb_dim'),
            bases=config.get('bases'),
            enrich_flag=config.get('enrich_flag')
        )

    elif model_name == 'rgcn_emb':
        model = RGCN_EMB(
            triples=triples,
            num_nodes=num_nodes,
            num_rels=num_rels,
            num_classes=num_classes,
            emb_dim=config.get('emb_dim'),
            weights_size=config.get('weights_size'),
            bases=config.get('bases'),
            enrich_flag=config.get('enrich_flag')
        )

    elif model_name == 'lgcn':
        model = LGCN(
            triples=triples,
            num_nodes=num_nodes,
            num_rels=num_rels,
            num_classes=num_classes,
            emb_dim=config.get('emb_dim'),
            rp=config.get('rp'),
            ldepth=config.get('ldepth'),
            lwidth=config.get('lwidth'),
            enrich_flag=config.get('enrich_flag')
        )

    elif model_name == 'lgcn_rel_emb':
        model = LGCN_REL_EMB(
            triples=triples,
            num_nodes=num_nodes,
            num_rels=num_rels,
            num_classes=num_classes,
            emb_dim=config.get('emb_dim'),
            rp=config.get('rp'),
            enrich_flag=config.get('enrich_flag')
        )

    else:
        raise ValueError(f'Unknown model name: {model_name}')

    return model
