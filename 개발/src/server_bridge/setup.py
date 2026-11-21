from setuptools import setup

setup(
    name="server_bridge",
    version="0.1.0",
    packages=["server_bridge"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/server_bridge"]),
        ("share/server_bridge", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    description="MAICON server_bridge: game server HTTP <-> ROS",
    license="LicenseRef-Proprietary",
    maintainer="MAICON team",
    maintainer_email="team@example.invalid",
    entry_points={"console_scripts": ["server_bridge_node = server_bridge.node:main"]},
)
