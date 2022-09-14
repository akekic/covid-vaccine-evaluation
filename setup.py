from setuptools import setup, find_namespace_packages

with open("README.md") as f:
    long_description = f.read()


def parse_requirements(filename):
    """load requirements from a pip requirements file"""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


setup(
    name="covid-vaccine-policy",
    author="Armin Kekić",
    author_email="armin.kekic@mailbox.org",
    packages=find_namespace_packages(),
    url="https://github.com/akekic/covid-vaccine-policy",
    description="Simulation-assisted causal modeling for evaluating counterfactual COVID-19 vaccine allocations strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.9.0",
    install_requires=parse_requirements("./requirements.txt"),
)
