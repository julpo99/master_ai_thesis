OPTUNA_SEARCH_SPACE ={
'rgcn': {
        'lr': ('float', 0.01, 0.1, 'log'),
        'wd': ('float', 0.0001, 0.03, 'log'),
        'l2': ('float', 0.00001, 0.001, 'log'),
        'epochs': ('int', 30, 200),
        'optimizer': ('categorical', ['adam', 'adamw']),
        'emb_dim': ('int', 8, 64),
        'bases': ('categorical', [None] + list(range(1, 51))),
        'enrich_flag': ('categorical', [True, False])
    },

    'rgcn_emb': {
        'lr': ('float', 0.001, 0.1, 'log'),
        'wd': ('float', 0.00001, 0.01, 'log'),
        'l2': ('float', 0.00001, 0.001, 'log'),
        'epochs': ('int', 5, 60),
        'optimizer': ('categorical', ['adam', 'adamw']),
        'emb_dim': ('int', 512, 2048),
        'weights_size': ('int', 32, 128),
        'bases': ('categorical', [None] + list(range(1, 65))),
        'enrich_flag': ('categorical', [True, False])
    },

    'lgcn': {
        'lr': ('float', 0.0001, 0.01, 'log'),
        'wd': ('float', 0.0001, 0.01, 'log'),
        'l2': ('float', 0.00001, 0.01, 'log'),
        'epochs': ('int', 150, 300),
        'optimizer': ('categorical', ['adamw']),
        'emb_dim': ('int', 8, 128),
        'rp': ('int', 1, 32),
        'ldepth': ('int', 0, 8),
        'lwidth': ('int', 64, 512),
        'bases': ('categorical', [None]),
        'dropout': ('float', 0.1, 0.5),
        'enrich_flag': ('categorical', [False])
    },

    'lgcn_rel_emb': {
        'lr': ('float', 0.0001, 0.1, 'log'),
        'wd': ('float', 0.000001, 1, 'log'),
        'l2': ('float', 0.0000001, 0.01, 'log'),
        'epochs': ('int', 30, 250),
        'optimizer': ('categorical', ['adam', 'adamw']),
        'emb_dim': ('int', 8, 128),
        'rp': ('int', 1, 10),
        'bases': ('categorical', [None]),
        'dropout': ('float', 0.0, 0.4),
        'enrich_flag': ('categorical', [False])
    },
}