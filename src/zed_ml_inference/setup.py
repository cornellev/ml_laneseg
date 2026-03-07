from setuptools import find_packages, setup
import os
package_name = 'zed_ml_inference'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('lib', package_name), ['zed_ml_inference/model_epoch_150.pth']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cam-cev',
    maintainer_email='clm357@cornell.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_segmentation_node = zed_ml_inference.lane_segmentation_node:main',
            'test_model = zed_ml_inference.test_model:main',
            'test_camera = zed_ml_inference.test_camera:main',
        ],
    },
)
