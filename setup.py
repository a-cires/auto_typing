from skbuild import setup
from setuptools import find_packages

setup(
    name='auto_typing',
    version='0.1',
    packages=find_packages(where="auto_typing"),
    package_dir={"": "auto_typing"},
    include_package_data=True,
    install_requires=[
        'numpy',
        'opencv-python',
        'PyYAML',
        'pytesseract',
        'scikit-learn',
        'python-can',
    ],
    cmake_source_dir='cpp_ext',  # Points to your C++ extension
)
