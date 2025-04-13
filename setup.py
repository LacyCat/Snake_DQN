from setuptools import setup, find_packages

setup(
    name='snake-dqn',
    version='1.1.1',
    packages=find_packages(include=["snake_dqn_core", "snake_dqn_core.*"]),
    install_requires=[
        'torch>=2.0',
        'pygame>=2.1.0',
        'numpy',
        'gym',  # 필요하다면 추가
    ],
    author='LacyCat',
    description='A simple DQN implementation for Snake game',
    url='https://github.com/LacyCat/Snake_DQN',  # GitHub 링크 추가
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
