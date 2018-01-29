import argparse

import numpy as np
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--config_name', type=str, default="TripletMnist")

    torch.manual_seed(1137)
    np.random.seed(1137)
    from config import set_config, get_config

    args = parser.parse_args()
    set_config(args.config_name)
    import trainer as trainer_module

    trainer = getattr(trainer_module, get_config().trainer)

    print(get_config().__dict__)
    trainer.run()
