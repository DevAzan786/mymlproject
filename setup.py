from setuptools import setup, find_packages
from typing import List
def get_requirements(file_path:str)->List[str]:
   '''
    This function reads the requirements file and returns a list of requirements
   '''
   requirements = []
   with open(file_path, 'r') as file:
       for line in file:
           requirements.append(line.strip())
       if "-e ." in requirements:
           requirements.remove("-e .")
   return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Azan Ali',
    author_email='azanaliworks@gmail.com',
    packages = find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
