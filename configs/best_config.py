BEST_CONFIG = {
    'rgcn': {
        'model_name': 'rgcn',
        'lr': 0.05499010733725328,
        'wd': 0.00022666309964608877,
        'l2': 0.000514707867508644,
        'epochs': 163,
        'optimizer': 'adamw',
        'emb_dim': 31,
        'bases': None,
        'enrich_flag': True
    },
    'rgcn_emb': {
        'model_name': 'rgcn_emb',
        'lr': 0.0016366409775459736,
        'wd': 0.0005755564025828806,
        'l2': 4.854651125478105e-05,
        'epochs': 45,
        'optimizer': 'adam',
        'emb_dim': 1236,
        'weights_size': 49,
        'bases': 29,
        'enrich_flag': True
    },
    'lgcn': {
        'model_name': 'lgcn',
        'lr': 0.007098298759085652,
        'wd': 0.0001928055417451552,
        'l2': 0.0001700540299938808,
        'epochs': 157,
        'optimizer': 'adamw',
        'emb_dim': 88,
        'rp': 5,
        'ldepth': 1,
        'lwidth': 142,
        'enrich_flag': True
    },
    'lgcn_rel_emb': {
        'model_name': 'lgcn_rel_emb',
        'lr': 0.029164628546417427,
        'wd': 1.8972659752382112e-06,
        'l2': 2.0660993991588798e-07,
        'epochs': 384,
        'optimizer': 'adam',
        'emb_dim': 121,
        'rp': 2,
        'enrich_flag': True
    }
}
