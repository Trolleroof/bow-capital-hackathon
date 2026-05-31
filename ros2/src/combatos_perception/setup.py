from setuptools import find_packages, setup

package_name = "combatos_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/oak_yolox_perception.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="CombatOS",
    maintainer_email="vikram.kommera@gmail.com",
    description="ROS2 YOLO/YOLOX perception node for OAK-D camera streams.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "yolox_node = combatos_perception.yolox_node:main",
        ],
    },
)
