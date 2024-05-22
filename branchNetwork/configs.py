"""Base Config for models"""
# __inti__.py

BASE_CONFIG = {'model_name': 'BranchModel', 
                   'repeat':1,
                   'n_in': 784, 
                    'n_out': 10, 
                    'n_contexts': 1, 
                    'device': 'cpu', 
                    'n_npb': [56, 56], 
                    'n_branches': [14, 14], 
                    'sparsity': 0.8,
                    'dropout': 0.5,
                    'hidden_layers': [784, 784],
                    'lr': 0.0001,
                    'batch_size': 32,
                    'epochs_per_train': 20,
                    'permute_seeds': [None, 21, 42],
                    'rotations': [0],
                    'device': 'cpu',
                    'learn_gates': False,
                    }