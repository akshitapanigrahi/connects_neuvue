from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    # Filter out comments and empty lines
    return [line for line in lines if line and not line.startswith('#')]

setup(
    name='connects_neuvue',
    version='0.1.0',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    extras_require={
        'morph_tools': [
            'neuron_morphology_tools @ git+https://github.com/reimerlab/neuron_morphology_tools.git'
        ]
    },
    author='Brendan Celii',
    description='a package with API defined for querying connects data products for neuvue visualizations and other downstream tasks',
    url='https://github.com/reimerlab/connects_neuvue',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
