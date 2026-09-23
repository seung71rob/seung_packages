from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'robot_description'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'urdf'),
            glob('urdf/*.urdf')
        ),
        (
            os.path.join('share', package_name, 'meshes_robot'),
            glob('meshes_robot/*')
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seung',
    maintainer_email='seung@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'robot_arm_node = robot_description.robot_arm_node:main',
            'robot_con = robot_description.robot_con:main',
            'robot_ud = robot_description.robot_ud:main',
            'robot_arm_node_rpm = robot_description.robot_arm_node_rpm:main',
            'robot_ud_vel = robot_description.robot_ud_vel:main',
            'robot_arm_node_pos_rpm = robot_description.robot_arm_node_pos_rpm:main',
            'robot_ud_pos_rpm = robot_description.robot_ud_pos_rpm:main',
            'robot_ik_node = robot_description.robot_ik_node:main',
            'robot_arm_view_node = robot_description.robot_arm_view_node:main',
            'robot_target_cmd = robot_description.robot_target_cmd:main',
            'ik_pybullet_node = robot_description.ik_pybullet_node:main',
            'ik_fix_pybullet_node = robot_description.ik_fix_pybullet_node:main',
            'ik_pybullet_node_5axis = robot_description.ik_pybullet_node_5axis:main',
            'ik_path_node_5axis = robot_description.ik_path_node_5axis:main',
            'ik_execute_node_5axis = robot_description.ik_execute_node_5axis:main',
            'path_collision_guard_5axis = robot_description.path_collision_guard_5axis:main',
            'motion_execute_node_5axis = robot_description.motion_execute_node_5axis:main',
            'ik_execute_centric_5axis = robot_description.ik_execute_centric_5axis:main',
            'ik_execute_suction_5axis = robot_description.ik_execute_suction_5axis:main',
            'ik_execute_rail_centric_5axis = robot_description.ik_execute_rail_centric_5axis:main',
            'ik_execute_rail_suction_5axis = robot_description.ik_execute_rail_suction_5axis:main',
            'ik_execute_rail_vision_centric_5axis = robot_description.ik_execute_rail_vision_centric_5axis:main',
            'ik_execute_rail_vision_suction_5axis = robot_description.ik_execute_rail_vision_suction_5axis:main',
            'process_sequence_5axis = robot_description.process_sequence_5axis:main',
            'centric_sequence_5axis = robot_description.centric_sequence_5axis:main',
            'suction_sequence_5axis = robot_description.suction_sequence_5axis:main',
        ],
    },
)
