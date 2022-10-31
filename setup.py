
  
from setuptools import setup, find_packages

setup(
    name='is_weapons_detector',
    version='0.1.0',
    description='',
    url='http://github.com/projeto-videomonitoramento/is-weapons-detector',
    author='labvisio',
    license='MIT',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    entry_points={
        'console_scripts': [
            'is-weapons-detector=is_weapons_detector.main:main',
        ],
    },
    zip_safe=False,
    install_requires=[
        'is-wire==1.2.0',
        'is-msgs==1.1.11',
        'numpy==1.18.5',
        'opencv-python==4.5.4.60',
        'opencensus-ext-zipkin==0.2.1',
        'python-dateutil==2.8.0',
        'vine==1.3.0',
        'torch==1.10.0',
        'yolov5==6.0.4',
    ],
)