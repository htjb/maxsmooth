from setuptools import setup, find_packages


def readme(short=False):
    with open('README.rst') as f:
        if short:
            return f.readlines()[1].strip()
        else:
            return f.read()

setup(
    name='maxsmooth',
    version='1.0.0',
    description='maxsmooth:Derivative Constrained Function Fitting',
    long_description=readme(),
    author='Harry T. J. Bevins',
    author_email='htjb2@cam.ac.uk',
    url='https://github.com/htjb/maxsmooth',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'cvxopt', 'matplotlib', 'progressbar'],
    license='MIT',
    extras_require={
          'docs': ['sphinx', 'sphinx_rtd_theme', 'numpydoc'],
          },
    tests_require=['pytest'],
    classifiers=[
               'Intended Audience :: Science/Research',
               'Natural Language :: English',
               'License :: OSI Approved :: MIT License',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8',
               'Topic :: Scientific/Engineering',
               'Topic :: Scientific/Engineering :: Astronomy',
               'Topic :: Scientific/Engineering :: Physics',
    ],
)
