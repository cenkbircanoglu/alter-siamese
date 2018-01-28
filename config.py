import configs

Config = None


def set_config(config):
    global Config
    Config = getattr(configs, config)()


def get_config():
    global Config
    return Config
