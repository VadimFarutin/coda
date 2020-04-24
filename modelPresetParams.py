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
            'optimizer': 'adagrad',
            'lr': 1e-3
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
            'optimizer': 'adam',
            'lr': 1e-3
        },
    },
    'lstm': {
        'model_library': 'pytorch',
        'model_class': 'SeqToSeq',
        'model_type': 'lstm',
        'model_specific_params': {
            'hidden_size': 128,
            'num_layers': 4,
            'bidirectional': True,
            'dropout': 0.5
        },
        'compile_params': {
            'regression_loss': 'MSE',
            'class_loss': 'binary_crossentropy',
            'optimizer': 'adam',
            'lr': 1e-3            
        },
    },
    'encoder-decoder': {
        'model_library': 'pytorch',
        'model_class': 'SeqToSeq',
        'model_type': 'encoder-decoder',
        'model_specific_params': {
            'hidden_size': 16,
            'num_layers': 1,
            'bidirectional': True,
            'teacher_forcing': 0.0,
            'dropout': 0.6
        },
        'compile_params': {
            'regression_loss': 'MSE',
            'class_loss': 'binary_crossentropy',
            'optimizer': 'adam',
            'lr': 5e-4
        },
    },
    'cnn-encoder-decoder': {
        'model_library': 'pytorch',
        'model_class': 'SeqToSeq',
        'model_type': 'cnn-encoder-decoder',
        'model_specific_params': {
            'hidden_size': 4,
            'kernel_size': 51,
            'stride': 1,
            'dilation': 1,
            'num_layers': 5,
            'residual': True,
            'dropout': 0.0
        },
        'compile_params': {
            'regression_loss': 'MSE',
            'class_loss': 'binary_crossentropy',
            'optimizer': 'adam',
            'lr': 5e-4
        },
    },
    'adv-cnn-encoder-decoder': {
        'model_library': 'pytorch',
        'model_class': 'SeqToSeq',
        'model_type': 'adv-cnn-encoder-decoder',
        'model_specific_params': {
            'hidden_size': 4,
            'kernel_size': 51,
            'stride': 1,
            'dilation': 2,
            'num_layers': 4,
            'residual': True,
            'dropout': 0.0,

            'disc_hidden_size': 16,
            'disc_kernel_size': 1001,
            'disc_dilation': 1,
            'disc_num_layers': 1            
        },
        'compile_params': {
            'regression_loss': 'MSE',
            'class_loss': 'binary_crossentropy',
            'optimizer': 'adam',
            'lr': 1e-4
        },
    },
}
