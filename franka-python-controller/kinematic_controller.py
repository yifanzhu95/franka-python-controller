from collections import deque
import math
import os
from threading import Thread, Lock
import time
from typing import List, Sequence
import struct
import sys

import klampt
try:
    from motionlib import so3, se3, vectorops as vo
except ImportError:
    from klampt.math import so3, se3, vectorops as vo
from klampt.model import ik
import numpy as np
import cvxpy as cp


from .abstract_controller import track_methods, include_method
from .franka_kinematic_controller import KlamptModelController, ControlMode

import franka_motion

IK_FAIL_BUFFER_SIZE = 20
IK_FAIL_THRESHOLD = 15

@track_methods
class KinematicFrankaController(KlamptModelController):
    """Class for running franka robot control logic.

    Works in kinematic and physical mode. Responsible for stuff like IK heuristics
    """

    def __init__(self, name, robot_model, EE_link, collision_checker, params):
        """
        params arguments:

        - elbow_lookahead: Scan step when doing elbow optimization
        - elbow_speed: Scan step multiplier when doing elbow optimization (should be 1 tbh)
        - qmin: software joint limits (min)
        - qmax: software joint limits (max)
        - kinematic_home: Home config in kinematic mode (since all 0s is in collision)
        """
        super().__init__(name, robot_model, EE_link, collision_checker, params)

        # Janky: Reading link 0, assumed to be UR base link
        self.base_transform = robot_model.link("base_link").getTransform()
        self.shoulder_pos = robot_model.link("panda_link1").getTransform()[1]
        print(name, self.shoulder_pos, self.base_transform)

        self.elbow_link = robot_model.link("elbow_link")
        self.measured_elbow_transform = None

        self.min_drivers = np.array(params.get('qmin', self.qmin))
        self.max_drivers = np.array(params.get('qmax', self.qmax))

        # Kinematic simulation
        self.step_config = params.get('kinematic_home', [0.0]*7)
        robot_model.setConfig(robot_model.configFromDrivers(self.step_config))

        self.IK_fail_flag = False           # True if previous IK solve attempt failed
        self.IK_fail_count = 0
        self.IK_fail_buffer = deque([0]*IK_FAIL_BUFFER_SIZE)
        self.end_of_travel_flag = False     # Flag for being close to max extension (measured by elbow angle).
        self.self_collision_flag = False    # Not used in kinematic mode; used in physical to record self collision stops

    def _update_cvx_constraints(self, new_constraints):
        """Update convex problem constraints.

        Parameters:
        --------------------
            new_constraints:    Array[bool, 6]      constraint status to set (x, y, z, r, p, y)
        """
        change_happened = sum(np.logical_xor(self._active_constraints, new_constraints) > 0)
        if change_happened:
            self._reproject = sum(new_constraints) > 0
            self._active_constraints = new_constraints

            constraints = [self.min_drivers <= self._q + self._dq,
                           self.max_drivers >= self._q + self._dq]

            J_prod = (self._J @ self._dq)
            # Jacobian order is [rx, ry, rz, x, y, z].
            jac_order = [3, 4, 5, 0, 1, 2]
            for i, b in enumerate(new_constraints):
                if b:
                    constraints.append(J_prod[jac_order[i]] == 0)
            self._cvx_problem = cp.Problem(self._objective, constraints)

    def beginStep(self) -> None:
        super().beginStep()
        self.end_of_travel_flag = bool(self.measured_config[3] > -1.1)  # STUPID thing since json can't serialize _bool dumb
        self.measured_elbow_transform = self.elbow_link.getTransform()

    def drive_EE(self, target, params):
        """Compute the target joint configuration to send to the franka driver.
        based on a target end effector pose.

        Valid params:
            tool_center:    SE3     TCP transform relative to franka EE
            elbow:          Vec3    Target elbow location

        Parameters:
        --------------------
            target:         SE3     target end effector position from teleop.
            params:         dict    Other controller parameters, ex. tool center

        Return:
        --------------------
        (success, Union(config, None))
        """
        robot_model = self.klamptModel()
        hand_transform = target
        tool_offset = params.get('tool_center', se3.identity())
        target = se3.mul(target, se3.inv(tool_offset))
        R, t = target

        #m_bar = (0.1, 2)    # TODO: move to settings/tune
        # m_bar = 0.04    # TODO: move to settings/tune
        # repel_step = 0.001  # TODO: vector
        # R, t = self._singularity_avoidance(target, m_bar, repel_step, actives=list(range(1, 7)))

        goal = ik.objective(self.get_EE_link(), R=R, t=t)
        solver = ik.solver(goal, iters=100, tol=1e-5) #1e-3

        if solver.solve():
            cfg = robot_model.configToDrivers(robot_model.getConfig())
            return (True, cfg)

        return (False, None)

    def update_IK_failure(self, status: bool):
        self.IK_fail_buffer.append(int(status))
        self.IK_fail_count += status - self.IK_fail_buffer.popleft()
        self.IK_fail_flag = self.IK_fail_count > IK_FAIL_THRESHOLD

    def endStep(self) -> None:
        """Control the robot.

        In EE mode, attempts to pull the elbow towards
            a provided (or guessed) position in space.
        """
        robot_model = self.klamptModel()
        save_config = robot_model.getConfig()

        with self.control_lock:
            control_mode = self.control_mode
            target = self.target
            params = self.controller_params

        if control_mode == ControlMode.POSITION:
            self.step_config = target
            self.update_IK_failure(False)

        elif control_mode == ControlMode.POSITION_EE:
            success, cfg = self.drive_EE(target, params)
            self.update_IK_failure(not success)
            if success:
                self.step_config = cfg
            else:
                pass
                #print("ik solve fail", solver.getResidual(), vo.norm(solver.getResidual()), solver.getSecondaryResidual())
        else:
            self.update_IK_failure(False)

        robot_model.setConfig(save_config)

    @include_method
    def to_dict(self):
        with self.control_lock:
            ret = super().to_dict()
            ret['flags'] = {
                'end_of_travel': self.end_of_travel_flag,
                'IK_fail': self.IK_fail_flag,
                'self_collision': self.self_collision_flag
            }
            #print([f"{x}: {type(ret['flags'][x])}" for x in ret['flags']])
        return ret
