from setuptools import setup

setup(
    name="llm_worker",
    version="0.1.0",
    packages=["llm_worker"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/llm_worker"]),
        ("share/llm_worker", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    description="MAICON llm_worker: tactical order parsing",
    license="LicenseRef-Proprietary",
    maintainer="MAICON team",
    maintainer_email="team@example.invalid",
    entry_points={"console_scripts": ["llm_worker_node = llm_worker.node:main"]},
)
