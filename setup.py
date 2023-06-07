from setuptools import find_packages, setup

setup(
    name="m3drefclip",
    py_modules=["m3drefclip"],
    version="1.0",
    author="Yiming Zhang",
    description="M3DRef-CLIP",
    packages=find_packages(include=("m3drefclip*")),
    install_requires=[
        f"clip @ git+ssh://git@github.com/eamonn-zh/CLIP.git", "lightning", "wandb", "scipy", "hydra-core",
        "h5py", "open3d", "pandas"
    ]
)
