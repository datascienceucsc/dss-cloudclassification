try:  # for pip >= 10
    from pip._internal.req import parse_requirements
    from pip._internal.download import PipSession
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements
    from pip.download import PipSession

from os import path
from setuptools import setup, find_packages

# intall requirements from requirements.txt file
requirement_file_location = path.join(
    path.dirname(__file__), "requirements.txt")
install_req_objs = parse_requirements(
    requirement_file_location,
    session=PipSession())
install_reqs = [str(r.req) for r in install_req_objs]

if __name__ == "__main__":
    setup(
        author="UCSC Data Science Club",
        classifiers=[
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Programming Language :: Python :: 3.6",
            "Development Status :: 1 - Planning",
            "Intended Audience :: Developers",
            "Natural Language :: English",
        ],
        description="UCSC Data Science Kaggle 'Understanding Clouds from Satellite Images' competition",
        license="MIT",
        install_requires=install_reqs,
        include_package_data=True,
        name="datascience_kaggle_clouds",
        packages=find_packages(include=["dss_kaggle_clouds"]),
        entry_points={"console_scripts": [
            "kaggle_clouds = dss_kaggle_clouds:run"]},
        url="https://github.com/datascienceslugs/dss-cloudclassification",
        version="0.1.0",
        zip_safe=False,
    )
