BEST_CONFIG = {

    # 0.85
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

    # 0.81
    'rgcn_emb': {
        'model_name': 'rgcn_emb',
        'lr': 0.0010013,
        'wd': 0.00044336,
        'l2': 0.000012801,
        'epochs': 49,
        'optimizer': 'adam',
        'emb_dim': 1333,
        'weights_size': 125,
        'bases': 17,
        'enrich_flag': True
    },

    # 0.68 not enriched
    # 'lgcn': {
    #     'model_name': 'lgcn',
    #     'lr': 0.07701,
    #     'wd': 0.000011527,
    #     'l2': 0.0000024205,
    #     'epochs': 132,
    #     'optimizer': 'adamw',
    #     'emb_dim': 120,
    #     'rp': 10,
    #     'ldepth': 0,
    #     'lwidth': 240,
    #     'dropout': 0.0,
    #     'enrich_flag': False
    # },

    # # 0.68
    # 'lgcn': {
    #     'model_name': 'lgcn',
    #     'lr': 0.0070983,
    #     'wd': 0.00019281,
    #     'l2': 0.00017005,
    #     'epochs': 157,
    #     'optimizer': 'adam',
    #     'emb_dim': 88,
    #     'rp': 5,
    #     'ldepth': 1,
    #     'lwidth': 142,
    #     'dropout': 0.0,
    #     'enrich_flag': True
    # },
#     # trying to increase rp and ldepth
#     'lgcn': {
#         'model_name': 'lgcn',
#         'lr': 0.0070983,
#         'wd': 0.00019281,
#         'l2': 0.00017005,
#         'epochs': 157,
#         'optimizer': 'adam',
#         'emb_dim': 88,
#         'rp': 10,
#         'ldepth': 2,
#         'lwidth': 256,
#         'dropout': 0.1,
#         'enrich_flag': True
# },
#      'lgcn': {
#             'model_name': 'lgcn',
#             'lr': 0.0025,
#             'wd': 0.0001,
#             'l2': 0.0001,
#             'epochs': 500,
#             'optimizer': 'adam',
#             'emb_dim': 128,
#             'rp': 16,
#             'ldepth': 3,
#             'lwidth': 512,
#             'dropout': 0.1,
#             'enrich_flag': True
#     },
    #trying
    'lgcn': {
                'model_name': 'lgcn',
                'lr': 0.0066497014929140065,
                'wd': 0.0005877179632423122,
                'l2': 6.732669121389249e-05,
                'epochs': 300,
                'optimizer': 'adamw',
                'emb_dim': 16,
                'rp': 3,
                'ldepth': 4,
                'lwidth': 450,
                'dropout': 0.1315980673522603,
                'enrich_flag': True
        },

    # # 0.64
    # 'lgcn_rel_emb': {
    #     'model_name': 'lgcn_rel_emb',
    #     'lr': 0.0099831,
    #     'wd': 0.00017015,
    #     'l2': 0.000011562,
    #     'epochs': 240,
    #     'optimizer': 'adamw',
    #     'emb_dim': 126,
    #     'rp': 12,
    #     'enrich_flag': False
    # },

    # # 0.74
    # 'lgcn_rel_emb': {
    #         'model_name': 'lgcn_rel_emb',
    #         'lr': 0.029165,
    #         'wd': 0.0000018973,
    #         'l2': 2.0661e-7,
    #         'epochs': 384,
    #         'optimizer': 'adamw',
    #         'emb_dim': 64,
    #         'rp': 2,
    #         'dropout': 0.0,
    #         'enrich_flag': False
    #     },

    # trying
    'lgcn_rel_emb': {
            'model_name': 'lgcn_rel_emb',
            'lr': 0.005,
            'wd': 0.001,
            'l2': 0.0,
            'epochs': 300,
            'optimizer': 'adamw',
            'emb_dim': 16,
            'rp': 33,
            'dropout': 0.1,
            'enrich_flag': True
        },
}
