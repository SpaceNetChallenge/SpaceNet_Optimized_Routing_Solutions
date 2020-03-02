import os
import yaml


def get_config(path='configs/train_default.yml'):
    with open(path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def get_stage_templates(path='configs/stage_templates.yml'):
    with open(path, 'r') as stream:
        templates = yaml.safe_load(stream)
    return templates


def get_paths(path='configs/paths_default.yml'):

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, 'r') as stream:
        data_config = yaml.safe_load(stream)
    return data_config


if __name__ == '__main__':
    print(get_paths()['weights'])

    train_config = get_config()
    print(train_config)
    stages = train_config['stages']

    for stage in stages:
        stage_conf = stages[stage]
        print(stage_conf)
