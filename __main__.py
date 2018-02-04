import argparse

import numpy as np
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('trainer')
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--channel', type=int)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--network', type=str)
    parser.add_argument('--embedding', type=int)
    parser.add_argument('--loss', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--negative', type=int, default=0)
    parser.add_argument('--positive', type=int, default=1)

    torch.manual_seed(1137)
    np.random.seed(1137)
    from config import set_config, get_config

    args = parser.parse_args()

    kwargs = vars(args)
    trainer_name = kwargs['trainer']
    kwargs.pop('trainer')

    set_config(trainer_name, **kwargs)
    import trainer as trainer_module

    trainer = getattr(trainer_module, trainer_name)

    print(get_config().__dict__)
    trainer.run()
