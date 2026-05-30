from setuptools import find_packages, setup

package_name = "combatos_slam_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="CombatOS",
    maintainer_email="vikram.kommera@gmail.com",
    description="ROS2 to CombatOS WebSocket bridge for ORB-SLAM3 streams.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "slam_bridge = combatos_slam_bridge.slam_bridge:main",
        ],
    },
)
