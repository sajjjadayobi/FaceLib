import setuptools

with open("README.md", "r") as fh:
   long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

   
setuptools.setup(
    name='facelib',
    version='1.6',
    author="Sajjad Ayoubi",
    author_email="sadeveloper360@gmail.com",
    description="Face Detection & Age Gender & Expression & Recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sajjjadayobi/FaceLib",
    install_requires=required,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
