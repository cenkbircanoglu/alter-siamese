import configs

Config = None


def set_config(trainer, **kwargs):
    global Config
    Config = getattr(configs, trainer)(**kwargs)


def get_config():
    global Config
    return Config
