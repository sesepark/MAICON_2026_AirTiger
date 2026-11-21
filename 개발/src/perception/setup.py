from glob import glob

from setuptools import setup

setup(
    name="perception",
    version="0.1.0",
    packages=["perception"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/perception"]),
        ("share/perception", ["package.xml"]),
        ("share/perception/models", glob("training/drone_boxes/trained/*.pt")),  # git 에 넣은 드론 지도 상자 모델 (drone_box_model.bundled_model)
    ],
    install_requires=["setuptools"],
    description="MAICON perception: picture classification service for supervisor",
    license="LicenseRef-Proprietary",
    maintainer="MAICON team",
    maintainer_email="team@example.invalid",
    entry_points={"console_scripts": [
        "perception_node = perception.node:main",
    ]},
)
