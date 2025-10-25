from setuptools import find_packages, setup

package_name = 'ejercicio1_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cossio',
    maintainer_email='adriel.cossio@ucb.edu.bo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'joint_state_publisher = ejercicio1_pkg.joint_state_publisher:main',
            'joint_state_subscriber = ejercicio1_pkg.joint_state_subscriber:main'  ,
            'brazo_inverse_kinematics = ejercicio1_pkg.brazo_inverse_kinematics:main'
        ],
    },
)
