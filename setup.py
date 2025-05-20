"""
Setup configuration for the thermal model solver package.
"""

from setuptools import setup, find_packages

setup(
    name="arm_thermal_model_v2",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "json5",
    ],
    author="Menger Chen",
    description="Thermal model solver for robotic arm thermal analysis",
    python_requires=">=3.7",
) 