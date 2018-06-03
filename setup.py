from setuptools import setup

setup(
    name='keras-han-for-docla',
    version='0.1.0',
    packages=['keras_han'],
    url='',
    license='MIT',
    author='Floris Hoogenboom',
    author_email='floris@digitaldreamworks.nl',
    description='An inplementation of Hierarchical Attention Networks in Keras.',
    install_requires=[
        'keras>=2.1.5'
    ],
    test_requires=[
        'nose>=1.3.7'
    ]
)
