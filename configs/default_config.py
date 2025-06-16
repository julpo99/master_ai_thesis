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
        'wd': 0.0,
        'l2': 0.00,
        'epochs': 200,
        'optimizer': 'adam',
        'emb_dim': 128,
        'rp': 8,
        'ldepth': 1,
        'lwidth': 128,
        'enrich_flag': True
    },
    'lgcn_rel_emb': {
        'model_name': 'lgcn_rel_emb',
        'lr': 0.001,
        'wd': 0.0,
        'l2': 0.00,
        'epochs': 200,
        'optimizer': 'adam',
        'emb_dim': 128,
        'rp': 16,
        'enrich_flag': True
    },
    'lgcn_rel_emb_2': {
        'model_name': 'lgcn_rel_emb_2',
        'lr': 0.02,
        'wd': 0.0,
        'l2': 0.00001,
        'epochs': 200,
        'optimizer': 'adam',
        'emb_dim': 64,
        'rp': 16,
        'ldepth': 1,
        'lwidth': 64,
        'bases': None,
        'enrich_flag': True
    }
}