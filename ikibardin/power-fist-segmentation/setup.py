from setuptools import setup, find_packages

setup(
    name='power-fist',
    version='0.2.4',
    author='Ilya Kibardin',
    author_email='ikibardin@gmail.com',
    description='A flexible and advanced training pipeline',
    packages=find_packages(),
    install_requires=[
        'numpy >= 1.14.6',
        'torch >= 1.0',
        'torchvision >= 0.2.2',
        'opencv_python >= 4.0.0.21',
        'albumentations >= 0.2.3',
        # 'easygold >= 0.4.0',
        'pandas >= 0.23.4',
        'tqdm >= 4.28.1',
        'tensorboardX >= 1.6',
        'PyYAML >= 3.13',
        'pretrainedmodels >= 0.7.4',
        # 'apex >= 0.1',
    ],
)
