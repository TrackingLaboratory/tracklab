from setuptools import setup,  find_packages 

setup(
    name='posetrack21_mot',
    version='0.2',
    packages=find_packages(),
    install_requires=[
            'shapely',
            'numpy',
            'pandas',
            'scipy',
            'xmltodict',
            'tqdm',
            'lap',
    ],
)
