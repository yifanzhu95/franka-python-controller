"""
Base class for controllers that have their state fully reflected in the klampt model.
"""
from enum import Enum
import itertools
import os
import sys
from threading import Lock
from typing import Optional, List, Sequence

import numpy as np

from klampt.math import vectorops as vo
from klampt.math import se3, so3
from klampt.model import ik

from .abstract_controller import AbstractController, track_methods, include_method, tag_method
from .motionUtils import bring_in_limits

class ControlMode(Enum):
    NONE = -1
    POSITION = 0
    POSITION_EE = 1
    VELOCITY = 2
    VELOCITY_EE = 3
    FREEDRIVE = 4

@track_methods(filters=['requires_EE'])
class KlamptModelController(AbstractController):
    """
    Class representing any limb (but in kinematic mode only).
    """
    def __init__(self, name, robot_model, EE_link, collision_checker, params):
        super().__init__(name, robot_model, EE_link, collision_checker, {})
        self._n_drivers = robot_model.numDrivers()

        # NOTE: these are dof indices in the subrobot.
        # flatten getAffectedLinks() list of affected links.
        self.drivers = [robot_model.driver(i) for i in range(self._n_drivers)]
        self.driven_dofs = list(itertools.chain.from_iterable(driver.getAffectedLinks() for driver in self.drivers))
        self.paused = False
        self.controller_params: dict = {}
        self.target = None # Type: "any" (config/velocity/transform)
        self.control_mode = ControlMode.NONE
        self.control_lock = Lock()
        self.measured_config = None
        self.measured_EE_transform = None
        self.measured_vel = [0.0]*self.num_drivers()
        self.measured_EE_vel = [0.0]*6  # Twist vector. [rx, ry, rz, vx, vy, vz]
        self.measured_joint_torque = [0.0]*self.num_drivers()
        self.dt = params.get('dt', None)
        self.length_mismatch_warn = False

        # TODO: velocity limit
        # TODO: this is applied on config level when it should be applied at driver level...
        qmin, qmax = robot_model.getJointLimits()
        self.qmin = np.array(robot_model.configToDrivers(qmin))
        self.qmax = np.array(robot_model.configToDrivers(qmax))
        self.step_config = [0.0]*self.num_drivers()

    def get_api(self) -> List[str]:
        """EE-based methods will"""
        if self._api_cache is None:
            all_tracked = list(self.get_tracked_methods())
            if self._EE_link is not None:
                clazz = KlamptModelController
                all_tracked += [d[0] for d in clazz._tracker_tag_table['requires_EE']]
            self._api_cache = all_tracked
        return self._api_cache

    def num_drivers(self):
        return self._n_drivers

    def check_collision(self, _q1, _q2):
        """Use driven dofs instead of klampt dofs."""
        q1 = self.klamptModel().getConfig()
        q2 = q1[:]
        for i, n in enumerate(self.driven_dofs):
            q1[n] = _q1[i]
            q2[n] = _q2[i]
        return super().check_collision(q1, q2)

    def initialize(self) -> bool:
        """Kinematic limb controller has no init."""
        return True

    def close(self) -> bool:
        """Kinematic limb controller has no shutdown."""
        return True

    @include_method
    def status(self, joint_idx: Optional[int] = None) -> str:
        """Kinematic limb controller is just OK (or paused)."""
        if self.paused:
            return "paused"
        return "ok"

    @include_method
    def softStop(self):
        self.set_joint_velocity([0.0]*self.num_drivers(), {})
        self.paused = True
        self.control_mode = ControlMode.NONE

    @include_method
    def resume(self):
        self.paused = False

    def beginStep(self) -> None:
        # TODO: more realistic simulation and joint limits
        # TODO: velocity limits
        robot_model = self.klamptModel()
        old_config = robot_model.configToDrivers(robot_model.getConfig())
        self.measured_config = self.step_config
        self.measured_vel = (np.array(self.measured_config) - old_config) / self.dt
        robot_model.setConfig(robot_model.configFromDrivers(self.measured_config))

        if self.get_EE_link() is not None:
            measured_EE_transform = self.get_EE_link().getTransform()
            if self.measured_EE_transform is not None:
                old_transform = self.get_EE_transform()
                self.measured_EE_vel = vo.div(se3.error(measured_EE_transform, old_transform), self.dt)
            self.measured_EE_transform = measured_EE_transform

    def endStep(self) -> None:
        robot_model = self.klamptModel()
        save_config = robot_model.getConfig()

        with self.control_lock:
            control_mode = self.control_mode
            target = self.target
            params = self.controller_params

        target_config = self.step_config
        if control_mode == ControlMode.POSITION:
            target_config = target
        elif control_mode == ControlMode.POSITION_EE:
            if "tool_center" in self.controller_params:
                target = se3.mul(target, se3.inv(self.controller_params["tool_center"]))
            R, t = target
            goal = ik.objective(self.get_EE_link(), R=R, t=t)
            res = ik.solve_nearby(goal, iters=10, activeDofs=self.driven_dofs, maxDeviation=0.5)
            if res:
                target_config = robot_model.configToDrivers(robot_model.getConfig())
            else:
                print('IK solve failure: no IK solution found')
                print('motion.setEETransform({}):IK solve failure'.format(self.get_name()))
        elif control_mode == ControlMode.VELOCITY:
            target_config = vo.madd(self.step_config, target, self.dt)
        # Other control modes not supported for now.

        self.step_config = bring_in_limits(target_config, self.qmin, self.qmax)
        robot_model.setConfig(save_config)

    @include_method
    def set_joint_config(self, config: List[float], params: dict):
        if self.paused:
            return
        if len(config) != self.num_drivers():
            if not self.length_mismatch_warn:
                print(f"{self.get_name()}: Config length mismatch, not setting joint config")
                print("    This is expected for some components in kinematic mode.")
                self.length_mismatch_warn = True
            return

        with self.control_lock:
            self.target = config
            self.controller_params = params
            self.control_mode = ControlMode.POSITION

    @tag_method(requires_EE=True)
    def set_EE_transform(self, transform, params: dict):
        if self._EE_link is None:
            raise NotImplementedError(f"{self.get_name()}: set_EE_transform called on component with no EE link")
        if self.paused:
            return
        with self.control_lock:
            self.target = transform
            self.controller_params = params
            self.control_mode = ControlMode.POSITION_EE

    @include_method
    def set_joint_velocity(self, velocity: List[float], params: dict):
        if self.paused:
            return
        if len(velocity) != self.num_drivers():
            if not self.length_mismatch_warn:
                print(f"{self.get_name()}: Velocity length mismatch, not setting joint velocity")
                print("    This is expected for some components in kinematic mode.")
                self.length_mismatch_warn = True
            return

        with self.control_lock:
            self.target = velocity
            self.controller_params = params
            self.control_mode = ControlMode.VELOCITY

    @include_method
    def get_joint_config(self) -> List[float]:
        if self.measured_config is None:
            print(f"{self.get_name()}: WARNING: get_joint_config called, but no sensor data")
        return self.measured_config

    @tag_method(requires_EE=True)
    # def get_EE_transform(self, tool_center: Sequence[float] = (0.0, 0.0, 0.0)):
    def get_EE_transform(self, tool_center=None):
        """Get the end effector transform of this limb in the world frame.

        Return: Pair (R, T), 3x3 rotation matrix and 3d position of this limb's
                end effector, in the world frame (robot base frame)
        """
        if tool_center is None:
            tool_center = se3.identity()
        T = self.measured_EE_transform
        if self._EE_link is None:
            raise NotImplementedError(f"{self.get_name()}: get_EE_transform called on component with no EE link")
        if T is None:
            print(f"{self.get_name()}: WARNING: get_EE_transform called, but no sensor data")
            return None
        # return (T[0], vo.add(T[1], so3.apply(T[0], tool_center)))
        return se3.mul(T, tool_center)

    @tag_method(requires_EE=True)
    def get_EE_jacobian(self, toolcenter=(0, 0, 0)):
        # NOTE: THIS IS SO JANK, so much private variable usage!
        # Also subrobot's jacobian is broken anyway and I don't want it anyway
        if self._EE_link is None:
            raise NotImplementedError(f"{self.get_name()}: get_EE_jacobian called on component with no EE link")
        #dof_slice = np.array(self._robot_model._links)[self.driven_dofs] #this is for subrobot
        dof_slice = np.array(list(range(self._robot_model.numLinks())))[self.driven_dofs]
        # return np.vstack((self._EE_link._link.getOrientationJacobian()[:, dof_slice],
        #                   self._EE_link._link.getPositionJacobian(toolcenter)[:, dof_slice]))
        return np.vstack((self._EE_link.getOrientationJacobian()[:, dof_slice],
                    self._EE_link.getPositionJacobian(toolcenter)[:, dof_slice]))
        # Jac = np.array(self._EE_link.getJacobian((0,0,0)))
        # return Jac[:,[1,2,3,4,5,6]]
        #return 

    @tag_method(requires_EE=True)
    def get_EE_velocity(self) -> List[float]:
        if self._EE_link is None:
            raise NotImplementedError(f"{self.get_name()}: get_EE_velocity called on component with no EE link")
        return self.measured_EE_vel

    @include_method
    def get_joint_velocity(self) -> List[float]:
        return self.measured_vel

    @include_method
    def get_joint_torques(self) -> List[float]:
        return [0.0]*self.num_drivers()

    @include_method
    def to_dict(self):
        ret = super().to_dict()
        # HAHA race conditions!
        with self.control_lock:
            ret['command'] = {'type': self.control_mode._value_, 'target': self.target}
            ret['config'] = list(self.measured_config)
            ret['torques'] = list(self.measured_joint_torque)

        return ret
