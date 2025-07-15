DEFAULT_CONFIG = {
    'rgcn': {
        'model_name': 'rgcn',
        'lr': 0.01,
        'wd': 0.0,
        'l2': 0.0005,
        'epochs': 50,
        'optimizer': 'adam',
        'emb_dim': 16,
        'bases': None,
        'enrich_flag': False
    },
    'rgcn_emb': {
        'model_name': 'rgcn_emb',
        'lr': 0.01,
        'wd': 0.0,
        'l2': 0.0005,
        'epochs': 50,
        'optimizer': 'adam',
        'emb_dim': 1600,
        'weights_size': 16,
        'bases': None,
        'enrich_flag': True
    },
    'lgcn': {
        'model_name': 'lgcn',
        'lr': 0.001,
        'wd': 0.0005,
        'l2': 0.0,
        'epochs': 50,
        'optimizer': 'adamw',
        'emb_dim': 16,
        'rp': 33,
        'ldepth': 0,
        'lwidth': 128,
        'dropout': 0.0,
        'enrich_flag': False
    },
    'lgcn_rel_emb': {
        'model_name': 'lgcn_rel_emb',
        'lr': 0.01,
        'wd': 0.0,
        'l2': 0.0005,
        'epochs': 50,
        'optimizer': 'adam',
        'emb_dim': 16,
        'rp': 33,
        'dropout': 0.0,
        'enrich_flag': False
    },
}