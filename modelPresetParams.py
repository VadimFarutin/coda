MODEL_PRESET_PARAMS = {
    'cnn': {
        'model_library': 'keras',
        'model_class': 'SeqToPoint',
        'model_type': 'cnn',
        'model_specific_params': {
            'num_filters': 6,
            'filter_length': 51
        },
        'compile_params': {
            'regression_loss': 'MSE',
            'class_loss': 'binary_crossentropy',
            'optimizer': 'adagrad'
        },
    },
    'coda': {
        'model_library': 'pytorch',
        'model_class': 'SeqToPoint',
        'model_type': 'coda',
        'model_specific_params': {
            'num_filters': 6,
            'filter_length': 51
        },
        'compile_params': {
            'regression_loss': 'MSE',
            'class_loss': 'binary_crossentropy',
            'optimizer': 'adagrad'
        },
    },
    'atac': {
        'model_library': 'pytorch',
        'model_class': 'SeqToPoint',
        'model_type': 'atac',
        'model_specific_params': {
            'num_filters': 15,
            'filter_length': 51
        },
        'compile_params': {
            'regression_loss': 'MSE',
            'class_loss': 'binary_crossentropy',
            'optimizer': 'adam'
        },
    },
    'lstm': {
        'model_library': 'pytorch',
        'model_class': 'SeqToSeq',
        'model_type': 'lstm',
        'model_specific_params': {
            'hidden_size': 64,
            'num_layers': 1,
            'bidirectional': True,
            'dropout': 0.0
        },
        'compile_params': {
            'regression_loss': 'MSE',
            'class_loss': 'binary_crossentropy',
            'optimizer': 'adam'
        },
    },
}
