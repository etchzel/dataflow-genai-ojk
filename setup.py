import setuptools

setuptools.setup(
  name='modules',
  version='0.0.1',
  install_requires=[
    'apache-beam[gcp]==2.53.0',
    'google-cloud-aiplatform>=1.38'
  ],
  packages=setuptools.find_packages()
)