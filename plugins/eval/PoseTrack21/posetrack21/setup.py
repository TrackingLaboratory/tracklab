from setuptools import setup,  find_packages 

setup(
    name='posetrack21', 
    version='0.2',
    packages=find_packages(),
    install_requires=[
            'numpy',
            'Pillow',
            'pytest',
            'scipy',
            'Shapely',
            'sparse',
            'xmltodict',
    ],
)
