import setuptools

with open("README.md", "r") as fh:
   long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()


setuptools.setup(
    name='Cfacelib',
    version='0.1',
    author="lissettecarlr",
    author_email="lissettecarlr@gmail.com",
    description="Face Detection & Age Gender & Expression & Recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lissettecarlr/FaceLib",
    install_requires=required,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
