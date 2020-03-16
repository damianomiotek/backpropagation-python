import setuptools

# Get the long description from the README file.
with open('README.md') as f:
    _long_description = f.read()

setuptools.setup(
    name='backpropagation',
    version='0.1',
    description='Implementation of the backpropagation algorithm in python language',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/damianomiotek/backpropagation-python',
    author='Damian Omiotek',
    author_email='omiotekdamian@gmail.com',
    license='GNU GENERAL PUBLIC LICENSE',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'cycler',
        'kiwisolver',
        'pyparsing',
        'python-dateutil',
        'six'
    ],
    keywords='backpropagation python')
