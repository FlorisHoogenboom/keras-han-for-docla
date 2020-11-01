from setuptools import setup


TEST_DEPENDENCIES = [
    'pytest==6.1.2'
]

LINT_DEPENDENCIES = [
    'flake8==3.8.4'
]


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
        'keras>=2.1.5',
        'tensorflow>=2.2.0'
    ],
    test_requires=TEST_DEPENDENCIES,
    extras_require={
        'test': TEST_DEPENDENCIES,
        'lint': LINT_DEPENDENCIES,
        'dev': TEST_DEPENDENCIES + LINT_DEPENDENCIES
    }
)
