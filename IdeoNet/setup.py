from setuptools import setup, find_packages

with open("README.md", "r") as f:
    readme = f.read()

requirements = [
    "torch>=1.2.0",
    "torchvision",
    "matplotlib",
    "numpy",
    "pandas",
    "sklearn",
    "pre-commit",
]

extras = {
    "examples": [
        "tensorflow>=2.0.0",
        "ray[tune]",
        "ray[debug]",
        "requests",
        "bayesian-optimization",
    ],
    "tests": ["pytest", "coverage", "pytest-cov"],
}

setup(
    name="ideonet",
    version="0.1.0",
    description="Library for simulating SNN based on PyTorch.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Leon",
    author_email="lhsc@163.com",
    packages=find_packages(exclude=("tests", "examples")),
    install_requires=requirements,
    extras_require=extras,
    python_requires=">=3.6.0",
)
