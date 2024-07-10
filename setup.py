from setuptools import setup, find_packages

setup(
    name='neural_network_from_scratch',
    version='0.1.0',
    description='Build neural network from scratch',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Mohamed Karim Ben Boubaker',
    author_email='karimbb2002@gmail.com',
    url='https://github.com/Med-Karim-Ben-Boubaker/Neural-Network-From-Scratch',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        # Add other dependencies here
    ],
    entry_points={
        'console_scripts': [
            'your_command=your_project.module:function',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)