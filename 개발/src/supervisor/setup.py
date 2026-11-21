from setuptools import setup

setup(
    name="supervisor",
    version="0.1.0",
    packages=["supervisor"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/supervisor"]),
        ("share/supervisor", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    description="MAICON supervisor: mission flow, face records and reports",
    license="LicenseRef-Proprietary",
    maintainer="MAICON team",
    maintainer_email="team@example.invalid",
    entry_points={"console_scripts": ["supervisor_node = supervisor.node:main"]},
)
