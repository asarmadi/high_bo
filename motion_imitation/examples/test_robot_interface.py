"""Test the C++ robot interface.

Follow the
"""
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from robot_interface import RobotInterface # pytype: disable=import-error

i = RobotInterface()
o = i.receive_observation()
