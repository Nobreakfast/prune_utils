import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
exec(open("prune_utils/version.py").read())

setuptools.setup(
    name="prune_utils",
    version=__version__,
    author="Haocheng Zhao",
    author_email="Haocheng.Zhao@hotmail.com",
    description="Useful toolbox for Unstructured Pruning in Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nobreakfast/prune_utils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["torch", "torchvision", "numpy", "thop", "tqdm"],
    python_requires=">=3.6",
)
