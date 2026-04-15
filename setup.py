from setuptools import find_packages, setup

setup(
    name="retailai-churn-intelligence",
    version="0.1.0",
    packages=find_packages(
        include=[
            "app",
            "app.*",
            "ml",
            "ml.*",
            "data",
            "data.*",
            "migrations",
            "migrations.*",
        ]
    ),
)
