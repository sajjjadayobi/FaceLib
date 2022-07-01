import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name='facelib-kala',
    version='1.0',
    author="lissettecarlr",
    author_email="lissettecarlr@gmail.com",
    description="Face Detection & Recognition",
    url="https://github.com/lissettecarlr/FaceLib",
    install_requires=required,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
