from enum import Enum
from functools import wraps
from threading import Thread
from types import FunctionType
from typing import List, Optional, Sequence, Union, Set
import klampt

from klampt.math import so3, se3
from klampt.math import vectorops as vo
from klampt.model import ik
from klampt.model.subrobot import SubRobotModel
from klampt.robotsim import RobotModel, RobotModelLink
from klampt.control.robotinterface import RobotInterfaceBase

import numpy as np

def track_methods(*args, **kwargs):
    """Decorate a class to track methods declared in the class, and all of its parent classes recursively.
    NOTE: will give wrong results if multiple parent classes have method tracking enabled, or declare the function
    `get_tracked_methods`. Use with caution!    (OR maybe we can fix it by not using super()... e)

    Defines the function:
    ```
    def get_tracked_methods(self) -> Set[str]:
        # Returns a set of methods in this class and its superclasses that were marked for tracking.
        # Methods that are marked in parent classes can be excluded from this class's set by marking
        # them for exclusion.
        # Caches results. No option to force recomputation for now.
    ```

    Also defines static class variables `_tracker_tag_table` and `_tracker_cache`.

    Usage:

    ```
    @track_methods(track_superclass=False)
    class Test():
        ...

        @include_method
        def test_meth1(self):
            ...


    test = Test()
    test.get_tracked_methods()
    >> {'test_meth1'}

    # Also respects inheritance.
    @track_methods
    class Test2(Test):
        ...

        @include_method
        def test_meth2(self):
            ...


    test2 = Test2()
    test2.get_tracked_methods()
    >> {'test_meth1', 'test_meth2'}
    ```

    Ordering of methods not guaranteed. (It is a set.)
    TBH xmlrpc export should be reworked to use this mechanism. It is more general and like 50% less janky.

    Parameters:
    arg list should be length 0 or length 1.
        - length 0: accepting kwargs (configure)
        - length 1: accepting class
    kwargs:
        - track_superclass: Whether this class has a (trackable) superclass or not. Default: True
        - filters: Set of keys to construct table entries with. Keys ['include', 'exclude'] are always used.

    @see include_method, exclude_method
    """
    if len(args) == 0:
        return lambda clazz: track_methods(clazz, **kwargs)
    clazz = args[0]
    has_superclass = kwargs.get('track_superclass', True)
    filters = kwargs.get('filters', [])
    tracking_tags = list(set(filters) | {'include', 'exclude'})

    # Enumerate and pick out previously tracked methods.
    tag_table = {tag: [] for tag in tracking_tags}
    for prop_name, prop_obj in clazz.__dict__.items():
        if isinstance(prop_obj, FunctionType):
            if '_tracker_tag_method_flag' not in prop_obj.__dict__:
                continue
            tags = prop_obj._tracker_tag_method_flag
            include = 'include' in tags
            exclude = 'exclude' in tags
            if exclude:
                if not has_superclass:
                    print(f"WARNING: method_tracker: method {clazz.__name__}.{prop_obj.__name__} tagged with exclusion tag, but no superclass.")
                if include:
                    print(f"WARNING: method_tracker: method {clazz.__name__}.{prop_obj.__name__} tagged with both inclusion and exclusion tags.")
            for tag in tracking_tags:
                if tag in tags:
                    tag_table[tag].append((prop_name, prop_obj))
    clazz._tracker_tag_table = tag_table
    clazz._tracker_cache = None

    # Create appropriate functions for getting tracked methods.
    if has_superclass:
        def get_tracked_methods(self) -> Set[str]:
            if clazz._tracker_cache is None:
                super_meths = super(clazz, self).get_tracked_methods()
                includes = frozenset(name for name, obj in clazz._tracker_tag_table['include'])
                excludes = frozenset(name for name, obj in clazz._tracker_tag_table['exclude'])
                clazz._tracker_cache = super_meths | includes - excludes
            return clazz._tracker_cache
    else:
        def get_tracked_methods(self) -> Set[str]:
            if clazz._tracker_cache is None:
                clazz._tracker_cache = frozenset(name for name, obj in clazz._tracker_tag_table['include'])
            return clazz._tracker_cache
    clazz.get_tracked_methods = get_tracked_methods
    return clazz


def include_method(f):
    """Tag a method for tracking using the `track_methods` decorator."""
    return tag_method(include=True)(f)

def exclude_method(f):
    """Tag a method to not be tracked when using the `track_methods` decorator."""
    return tag_method(exclude=True)(f)

def tag_method(**kwargs):
    """Attach some data to a method.
    Tagged methods are collected in a nice place.
    """
    def tag(f):
        if '_tracker_tag_method_flag' in f.__dict__:
            f._tracker_tag_method_flag.update(kwargs)
        else:
            f._tracker_tag_method_flag = kwargs
        return f
    return tag

@track_methods(track_superclass=False)
class AbstractController(RobotInterfaceBase):
    """
    Class that does nothing. Just specifies capabilities of a Controller.
    Tracking is used on all functions that are exposed through the motion API.
    """

    def __init__(self, name: str, robot_model: RobotModel,
                 EE_link: Union[str, int], collision_checker, params: dict):
        """
        Parameters:
        ----------------
        name: Human readable name
        robot_model: subrobot (just this robot's dofs)
        EE_link: The link name or index in the subrobot to consider the end effector
        active_dofs: dofs indices in global model (trina entire robot) that correspond
                     to the dofs in this arm, in the same order
        collision_checker: Object must implement a function
                           `check_collision(q1, q2, dofs)`
                           where `q1, q2` are the values corresponding to arm dofs,
                           and `dofs` is the indices in global robot
        params: Dict containing extra parameters
        """
        RobotInterfaceBase.__init__(self)
        self._name: str = name
        self._robot_model: RobotModel = robot_model

        if EE_link is not None:
            self._EE_link: RobotModelLink = robot_model.link(EE_link)
        else:
            self._EE_link: RobotModelLink = None
        self._collision_checker = collision_checker
        self._api_cache = None
        self._params = params

    def get_name(self):
        return self._name

    def get_api(self) -> List[str]:
        """Return a list of functions which motion clients can call on this component.
        Ex. Base will not have exported set_joint_config
        """
        if self._api_cache is None:
            self._api_cache = list(self.get_tracked_methods())
        return self._api_cache

    def get_EE_link(self):
        """Gets the end effector link (klampt object), or None."""
        return self._EE_link

    def num_drivers(self):
        """Get the number of dofs in this component. (How many things in setconfig)"""
        return 0

    def check_collision(self, q1, q2):
        """Thin wrapper around collision checker for subclasses."""
        return self._collision_checker.check_collision(self._robot_model, q1, q2)

    def klamptModel(self) -> Optional[Union[RobotModel, SubRobotModel]]:
        """
        Gets the robot model.
        """
        return self._robot_model

    def initialize(self) -> bool:
        """Start the limb.
        Starts the robot driver.

        Return: True if startup success, else False
        """
        raise NotImplementedError("start is not implemented")

    def initializer_thread(self, logger):
        """Convenience func to spawn the thread to call init."""
        def startup_wrapper(controller):
            logger.info(f"Starting component {controller.get_name()}")
            controller.initialize()
            logger.info(f"Started component {controller.get_name()}")
            logger.debug(f"{controller.get_name()}: controller API: {controller.get_api()}")
        return Thread(group=None, target=startup_wrapper,
            name=f"{self.get_name()}:init", args=(self,))

    def close(self) -> bool:
        """Stop execution.
        Kills the robot driver, etc etc.

        return value unspecified? presumably true on success.
        """
        raise NotImplementedError("stop is not implemented")

    def shutdown_thread(self, logger = None):
        """Convenience func to spawn the thread to call close."""
        def close_wrapper(controller):
            #logger.info(f"Shutting down component {controller.get_name()}")
            controller.close()
            #logger.info(f"Shut down component {controller.get_name()}")
        return Thread(group=None, target=close_wrapper,
            name=f"{self.get_name()}:init", args=(self,))

    def beginStep(self) -> None:
        """
        The promise of this function is that the robot model reflects the actual configuration
        of the robot after this function is called for all components, EXCEPT for the base
        (base should store config outside of the klampt robot since motion is robot base local.)

        This function should update motion's internal model of the robot using sensor data from the driver.
        DO NOT PERFORM IO! The driver thread/process should update a class variable with sensor data,
        which is then read (possibly with a lock) and used to update the robot model.
        """
        pass

    def endStep(self) -> None:
        """
        This function must not change the klampt robot model.
        All components' beginStep() functions are called before any component's endStep() function is called.
        This ensures a consistent robot model for each endStep() call.

        IK solves should happen here, and nowhere else. DO NOT PERFORM IK IN DRIVER THREADS!
        A common pattern:
        ```
        class MyController():

            __init__(...):
                self.target_q = [...]
                self.target_EE = [...]

            def set_EE_transform(...):
                self.target_EE = target

            def endStep(self):
                self.target_q = ik.solve(self.target_EE, ...)

            # Push target_q out to hardware (IO) in a separate thread.
        ```

        This function should flush commands (saved in set_XXX calls) out to the driver process/thread.
        DO NOT PERFORM IO! Instead, update a class variable with command information,
        which should then be read (possibly with a lock) by the driver.
        """
        pass

    def status(self, joint_idx: Optional[int] = None) -> str:
        """Get the status of the component.

        Return: 'power_off' for powered off
                'booting' for it is booting
                'idle' for booted but controller not running
                'ok' for controller running
                'protective_stop' for motion error condition
        """
        raise NotImplementedError("status getter is not implemented")

    def softStop(self):
        """Pause execution.
        Set the limb to hold its position and not respond to commands.
        Does nothing if the limb is already paused.
        """
        raise NotImplementedError("pause (softStop) is not implemented")

    def resume(self):
        """Resume execution.
        If the robot is paused, set it to respond to commands again.
        Does nothing if the limb is not paused.
        """
        raise NotImplementedError("resume is not implemented")

    def setPosition(self, q):   # Klampt interface. no params, likely to break.
        return self.set_joint_config(q, {})
    def set_joint_config(self, config: List[float], params: dict):
        """Set the limb to move to a joint config.
        DO NOT PERFORM IO! Set a local variable; perform IO with the driver in a separate loop.

        Parameters:
            config: Position in configuration space to move to.

            Takes additional keyword arguments (ex. gains) for different arms.
        """
        raise NotImplementedError("set_joint_config is not implemented")

    def setVelocity(self, q):   # Klampt interface. no params, likely to break.
        return self.set_joint_velocity(q, {})
    def set_joint_velocity(self, velocity: List[float], params: dict):
        """Set the limb to move at a certain velocity in joint space.
        DO NOT PERFORM IO! Set a local variable; perform IO with the driver in a separate loop.

        Parameters:
            velocity: velocity in configuration space to move at.

            Takes additional keyword arguments (ex. gains) for different arms.
        """
        raise NotImplementedError("set_joint_velocity is not implemented")

    def setTorque(self, t, params):
        """Send a joint torque command to the robot.
        DO NOT PERFORM IO! Set a local variable; perform IO with the driver in a separate loop.

        Parameters:
            t: torque to apply at each joint, in order
        """
        raise NotImplementedError("setTorque is not implemented")

    def set_EE_transform(self, transform, params):
        """Set the limb to move its end effector to a position in cartesian space.
        DO NOT PERFORM IO! Set a local variable; perform IO with the driver in a separate loop.

        Parameters:
            transform: Position in task space to move to.
                       3x3 rotation matrix and 3D translation. (R, T)

            Takes additional keyword arguments (ex. gains) for different arms.
        """
        model = self.klamptModel()
        save_config = model.getConfig()
        model.setConfig(self.get_joint_config())
        R, t = transform
        goal = ik.objective(self._EE_link, R=R, t=t)
        if ik.solve(goal):
            target_config = model.getConfig()
            self.set_joint_config(target_config, params)
        else:
            print('IK solve failure: no IK solution found')
            status = 'motion.setEETransform({}):IK solve failure'.format(self._name)
            print(status)
        model.setConfig(save_config)

    def set_EE_velocity(self, velocity: List[float], params):
        """Set the limb to move its end effector at a certain velocity in cartesian space.
        DO NOT PERFORM IO! Set a local variable; perform IO with the driver in a separate loop.

        Parameters:
            velocity: Velocity in task space to move at. (rx, ry, rz, vx, vy, vz)

            Takes additional keyword arguments (ex. gains) for different arms.
        """
        raise NotImplementedError("set_EE_velocity not implemented")

    def get_joint_config(self) -> List[float]:
        """Get the joint configuration of this limb.
        DO NOT PERFORM IO! Read a local variable; perform IO with the driver in a separate loop.

        Return: List of joint angles.
        """
        raise NotImplementedError("get_joint_config not implemented")

    def get_joint_velocity(self) -> List[float]:
        """Get the joint velocities of this limb.
        DO NOT PERFORM IO! Read a local variable; perform IO with the driver in a separate loop.

        Return: List of joint velocities.
        """
        raise NotImplementedError("get_joint_velocity not implemented")

    def get_joint_torques(self) -> List[float]:
        """Get the joint torques of this limb.
        DO NOT PERFORM IO! Read a local variable; perform IO with the driver in a separate loop.

        Return: List of joint torques.
        """
        raise NotImplementedError("get_joint_torques not implemented")

    def get_EE_transform(self, tool_center):
        """Get the end effector transform of this limb in the world frame.
        DO NOT PERFORM IO! Read a local variable; perform IO with the driver in a separate loop.

        Return: Pair (R, T), 3x3 rotation matrix and 3d position of this limb's
                end effector, in the world frame (robot base frame)
        """
        if tool_center is None:
            tool_center = se3.identity()
        T = self._EE_link.getTransform()
        # return (T[0], vo.add(T[1], so3.apply(T[0], tool_center)))
        return se3.mul(T, tool_center)

    def get_EE_jacobian(self, toolcenter=(0, 0, 0)):
        """Get the jacobian of the end effector, where the matrix's inputs are the limb active
        degrees of freedom, in order, and output is a 6-vector (Wx Wy Wz Vx Vy Vz).
        DO NOT PERFORM IO! Read a local variable; perform IO with the driver in a separate loop.

        Return:
        -------------
            Jacobian matrix (6x6), described above.
        """
        raise NotImplementedError("get_EE_jacobian not implemented")

    def get_EE_velocity(self) -> List[float]:
        """Get the end effector velocity of this limb in the world frame.
        DO NOT PERFORM IO! Read a local variable; perform IO with the driver in a separate loop.

        Return: [rx, ry, rz, vx, vy, vz] list representing the velocity in the robot base frame
        """
        raise NotImplementedError("get_EE_velocity not implemented")

    def get_EE_wrench(self) -> List[float]:
        """Get the wrench at the end effector.
        DO NOT PERFORM IO! Read a local variable; perform IO with the driver in a separate loop.

        NOTE: THIS DOESN'T THROW `NotImplementedError` BY DEFAULT BECAUSE MANY COMPONENTS DO NOT
        IMPLEMENT THIS FUNCTIONALITY. OVERRIDE IT YOURSELF!

        Return: [tx, ty, tz, fx, fy, fz] list representing the wrench in the robot base frame
        """
        return [0.0]*6

    def set_freedrive(self, freedrive_mode: bool):
        """Set this component to be in freedrive mode (or not).
        DO NOT PERFORM IO! Set a local variable; perform IO with the driver in a separate loop.

        NOTE: THIS DOESN'T THROW `NotImplementedError` BY DEFAULT BECAUSE MANY COMPONENTS DO NOT
        IMPLEMENT THIS FUNCTIONALITY. OVERRIDE IT YOURSELF!

        Parameters:
        -----
        freedrive_mode: True to be in freedrive, False otherwise.
        """
        pass

    @include_method
    def to_dict(self) -> dict:
        """Return some state dict to be used for logging/debugging.
        Does not have to serialize whole controller state.
        Help interface with nairen's gripper code.

        DO NOT PERFORM IO! Read a local variable; perform IO with the driver in a separate loop.
        """
        ret = { 'params': self._params }
        if self._EE_link is not None:
            ret.update({'EE_link': self._EE_link.getName()})
        return ret
