from setuptools import setup, find_packages

setup(
    name='my_custom_models',
    version='1.0',
    # 1. Tell Python that the root of your packages is the 'source' folder
    package_dir={'': 'source'}, 
    
    # 2. Tell find_packages to only look inside 'source' for the packages
    packages=find_packages(where='source'), 
)