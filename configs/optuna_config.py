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
        'lr': ('float', 0.0001, 0.07, 'log'),
        'wd': ('float', 0.0001, 0.003, 'log'),
        'l2': ('float', 0.000001, 0.0003, 'log'),
        'epochs': ('int', 90, 180),
        'optimizer': ('categorical', ['adam']),  # fixed optimizer
        'emb_dim': ('int', 8, 96),
        'rp': ('int', 1, 14),
        'ldepth': ('int', 1, 1),
        'lwidth': ('int', 16, 156),
        'bases': ('categorical', [None]),
        'enrich_flag': ('categorical', [True, False])
    },

    'lgcn_rel_emb': {
        'lr': ('float', 0.001, 0.07, 'log'),
        'wd': ('float', 0.00001, 0.015, 'log'),
        'l2': ('float', 0.00001, 0.005, 'log'),
        'epochs': ('int', 120, 250),
        'optimizer': ('categorical', ['adam']),
        'emb_dim': ('int', 16, 128),
        'rp': ('int', 1, 12),
        'bases': ('categorical', [None]),
        'enrich_flag': ('categorical', [True, False])
    },

    'lgcn_rel_emb_2': {
        'lr': ('float', 0.0005, 0.05, 'log'),
        'wd': ('float', 1e-5, 1, 'log'),
        'l2': ('float', 1e-7, 1e-4, 'log'),
        'epochs': ('int', 120, 600),
        'optimizer': ('categorical', ['adam', 'adamw']),
        'emb_dim': ('int', 32, 128),
        'rp': ('int', 1, 12),
        'ldepth': ('int', 1, 4),
        'lwidth': ('int', 16, 128),
        'bases': ('categorical', [None]),
        'enrich_flag': ('categorical', [True, False])
    },
}