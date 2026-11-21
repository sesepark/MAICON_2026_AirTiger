from setuptools import setup

setup(
    name="navigation",
    version="0.1.0",
    packages=["navigation", "navigation.drive"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/navigation"]),
        ("share/navigation", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    description="MAICON mapping node scaffold and driving node",
    license="LicenseRef-Proprietary",
    maintainer="MAICON team",
    maintainer_email="team@example.invalid",
    entry_points={"console_scripts": [
        "navigation_slam = navigation.slam_node:main",
        "navigation_drive = navigation.driving_node:main",
    ]},
)
