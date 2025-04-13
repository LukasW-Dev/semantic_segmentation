from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'semantic_segmentation'

def package_data(pkg_dir, sub_dir):
    """Generic function to find package_data for `pkg_dir` under `sub_dir`."""
    results = []
    for root, dirs, files in os.walk(os.path.join(pkg_dir, sub_dir)):
        rel_dir = os.path.relpath(root, pkg_dir)
        files = [os.path.join(rel_dir, f) for f in files]
        results.append((os.path.join('share', package_name, rel_dir), files))
    return results

setup(
    name=package_name,
    version='0.0.0',
    py_modules=['semantic_segmentation_node'],
    package_dir={'': 'scripts'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
      
        # Include launch file
        ('share/semantic_segmentation/launch', ['launch/semantic_segmentation.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lukas Wimmer',
    maintainer_email='wimmer1luk@gmail.com',
    description='Semantic segmentation node using MMsegmentation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'semantic_segmentation_node = semantic_segmentation_node:main',
        ],
    },
)
