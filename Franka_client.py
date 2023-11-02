from xmlrpc.client import ServerProxy
from threading import Thread, Lock
import threading
import time
import os
import numpy as np
from klampt.math import so3, se3, vectorops as vo

dirname = os.path.dirname(__file__)

class FrankaClient:
    def __init__(self, address = 'http://127.0.0.1:8080'):
        self.s = ServerProxy(address)
        #self.shut_down = False

    def initialize(self):
        self.s.initialize()

    def start(self):
        self.s.start()

    def shutdown(self):
        self.s.shutdown()

    def get_joint_config(self):
        return self.s.get_joint_config()

    def get_joint_velocity(self):
        return self.s.get_joint_velocity()

    def get_joint_torques(self):
        return self.s.get_joint_torques()

    def get_EE_transform(self, tool_center = se3.identity()):
        return self.s.get_EE_transform(tool_center)

    def get_EE_velocity(self):
        return self.s.get_EE_velocity()

    def get_EE_wrench(self):
        return self.s.get_EE_wrench()

    def set_joint_config(self, q):
        self.s.set_joint_config(q)

    def set_EE_transform(self, T):
        self.s.set_EE_transform(T)

    def set_EE_velocity(self, v):
        self.s.set_EE_velocity(v)

if __name__=="__main__":
    ft_arm_driver = FrankaClient('http://localhost:8080')
    ft_arm_driver.initialize()
    ft_arm_driver.start()

    print(ft_arm_driver.get_joint_config())

    ft_arm_driver.shutdown()

