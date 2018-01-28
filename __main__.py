import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--config_name', type=str, default="SiamCifar")
    from config import set_config, get_config

    args = parser.parse_args()
    set_config(args.config_name)

    from trainer import siamese_trainer

    print(get_config().__dict__)
    siamese_trainer.run()
