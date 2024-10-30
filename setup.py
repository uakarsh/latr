from setuptools import setup, find_packages

setup(
  name = 'latr',
  packages = find_packages(exclude=['examples']),
  version = '0.0.1',
  license='MIT',
  description = 'LaTr: Layout-aware transformer for scene-text VQA',
  author = 'Akarsh Upadhay',
  author_email = 'akarshupadhyayabc@gmail.com',
  url = 'https://github.com/uakarsh/latr',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'scene-text VQA',
  ],
  install_requires=[
    'torch>=1.6',
    'torchvision',
    'transformers',
    'sentencepiece==0.1.91',
    'Pillow==9.3.0'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ],
)
