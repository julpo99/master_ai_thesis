DEFAULT_CONFIG = {
    'rgcn': {
        'model_name': 'rgcn',
        'lr': 0.01,
        'wd': 0.0,
        'l2': 0.0005,
        'epochs': 50,
        'optimizer': 'adam',
        'emb_dim': 16,
        'bases': 40,
        'enrich_flag': True
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
        'epochs': 300,
        'optimizer': 'adamw',
        'emb_dim': 256,
        'rp': 32,
        'ldepth': 2,
        'lwidth': 128,
        'dropout': 0.0,
        'enrich_flag': True
    },
    'lgcn_rel_emb': {
        'model_name': 'lgcn_rel_emb',
        'lr': 0.01,
        'wd': 0.1,
        'l2': 0.001,
        'epochs': 400,
        'optimizer': 'adamw',
        'emb_dim': 64,
        'rp': 2,
        'dropout': 0.2,
        'enrich_flag': True
    },
}