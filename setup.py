from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['absl-py', 'keras>=2.2.4', 'colored_traceback', 'tensorboard']

setup(name='scope',
      version='0.1',
      description='Measure deep learning observables involving gradients '
                  'and Hessians',
      url='http://github.com/guygurari/scope',
      author='Guy Gur-Ari',
      author_email='guy@gurari.net',
      license='3-clause BSD',
      install_requires=REQUIRED_PACKAGES,
      packages=['scope'],
      zip_safe=False)
