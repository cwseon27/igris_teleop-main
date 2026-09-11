from glob import glob
import os

from setuptools import setup


package_name = "igris_reliability_runtime"


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name, "train_lib"],
    scripts=[
        "scripts/hand_confidence_inference",
        "scripts/controller_confidence_inference",
        "scripts/verify_policy_load",
        "scripts/hand_retarget_fusion",
    ],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config", "hand"), glob("config/hand/*.json")),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="seon",
    maintainer_email="seon@example.com",
    description="Reliability policy inference for IGRIS teleoperation.",
    license="MIT",
)
