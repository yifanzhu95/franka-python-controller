import copy
import ctypes
from enum import Enum
import math
import multiprocessing as mp
from typing import List
    
import time
import numpy as np
from klampt.model.trajectory import Trajectory
from klampt.model.subrobot import SubRobotModel

from klampt.math import vectorops as vo
from klampt.math import so2


class KinematicLimbController:
    def __init__(
        self,
        getConfig,
        getVelocity,
        setConfig,
        setVelocity,
        newState,
        limb_lower_limit,
        limb_upper_limit,
        limb_velocity_limit,
    ):
        """
        Parameters:
        -------------
        getConfig: Get limb config callback
        getVelocity: Get limb velocity callback
        setConfig: Set limb config callback
        setVelocity: Set limb velocity callback
        newState: newState update needed callback

        Note: All of these are just basically wrapping simulated_robot

        """
        self.type = "Kinematic"
        self.moving = None
        self.getConfig = getConfig
        self.getVelocity = getVelocity
        self.setConfig = setConfig
        self.setVelocity = setVelocity
        self.setFreeDrive = lambda self, active: None
        self.setFreeDrive = lambda mode: print(
            "Warning: KinematicLimbController has no freedrive option"
        )
        self.getCurrentTime = None

        def _setWrench(target, wrench_in, damping=None, task_frame=None):
            print("Warning: KinematicLimbController has no setWrench option.")
            self.setConfig(target)

        self.setWrench = _setWrench
        self.getWrench = lambda filtered: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.getJointTorques = lambda: [0.0] * 6
        self.getJointCurrents = lambda: [0.0] * 6
        self.get_current_error = lambda: [0.0] * 6
        self._joint_current_targets = [0.0] * 6
        self._joint_torque_targets = [0.0] * 6
        self.getOffset = lambda: [0.0] * 6
        self.get_cog = lambda: [0.0] * 3
        self.get_payload = lambda: 0.0
        self.zero_ft_sensor = lambda: None
        self.isProtectiveStopped = lambda: False

        self.markRead = lambda: None
        self.newState = newState

        self.qmin = limb_lower_limit
        self.qmax = limb_upper_limit
        self.vmin = [-x for x in limb_velocity_limit]
        self.vmax = limb_velocity_limit

        # Is this needed? idk
        self.start = lambda: True
        # This is needed.
        self.stop = lambda: True
        # Lol
        self.update = lambda: None


def clamp_limits(limb_lower_limit, limb_upper_limit, limb_velocity_limit,
    hw_limb_lower_limit, hw_limb_upper_limit, hw_limb_velocity_min,
    hw_lim_velocity_max
):
    limb_velocity_min = [-x for x in limb_velocity_limit]
    limb_velocity_max = [x for x in limb_velocity_limit]
    for i in range(len(limb_velocity_limit)):
        if hw_limb_lower_limit[i] > limb_lower_limit[i]:
            limb_lower_limit[i] = hw_limb_lower_limit[i]
        if hw_limb_upper_limit[i] < limb_upper_limit[i]:
            limb_upper_limit[i] = hw_limb_upper_limit[i]
        if hw_limb_velocity_min[i] > limb_velocity_min[i]:
            limb_velocity_min[i] = hw_limb_velocity_min[i]
        if hw_lim_velocity_max[i] < limb_velocity_max[i]:
            limb_velocity_max[i] = hw_lim_velocity_max[i]
    return (limb_lower_limit, limb_upper_limit, limb_velocity_min, limb_velocity_max)


class BaseControlMode(Enum):
    NOTHING = 0
    VELOCITY = 1
    PATH_FOLLOWING = 2
    VELOCITY_RAMPED = 3


class TorsoState:
    def __init__(self):
        self.measuredHeight = 0.0
        self.measuredTilt = 0.0
        self.commandedHeight = 0.0
        self.commandedTilt = 0.0

        self.commandSent = True
        self.leftLeg = 0
        self.rightLeg = 0

class HeadState:
    def __init__(self):
        self.sensedPosition = [0.0, 0.0]
        self.sensedVelocity = [0] * 2
        self.commandedPosition = [0.0, 0.0]
        self.newCommand = False


def bring_in_limits(q, min_limits, max_limits):
    """
    TODO: Do this on a different layer so that we can do collision checking properly...
    """
    return np.minimum(np.maximum(q, min_limits), max_limits)


def howFarOutOfLimits(q, min_limits, max_limits):
    err = 0
    for qi, a, b in zip(q, min_limits, max_limits):
        if qi < a:
            err = max(err, a - qi)
        if qi > b:
            err = max(err, qi - b)
    return err


def inLimits(q, min_limits, max_limits):
    """
    Check if a configuration is withing limits.
    Parameters:
        q:          Configuration to check.
        min_limits: Minimum joint limits. Default of None will auto-fail.
        max_limits: Maximum joint limits. Default of None will auto-fail.
    Return:
        True if every number `q[i]` is between `min_limits[i]` and `max_limits[i]`, inclusive; else False.
    """
    for qi, a, b in zip(q, min_limits, max_limits):
        if qi < a or qi > b:
            return False
    return True

def at_limits(q, min_limits, max_limits):
    """
    Find limits violations.
    Parameters:
        q:          Configuration to check.
        min_limits: Minimum joint limits. Default of None will auto-fail.
        max_limits: Maximum joint limits. Default of None will auto-fail.
    Return:
        List of failing indices
    """
    ret = []
    for i, qi, a, b in zip(range(6), q, min_limits, max_limits):
        if qi < a or qi > b:
            ret.append(i)
    return ret


def pinchness(q):
    """
    A measure of how close you are to pinch. Smaller is closer
    """
    y0 = 0.1250  # Joint 4 'd'... kinda... measured
    r2 = 0.0997  # Joint 5 'd'
    r1 = 0.0996  # Joint 6 'd'
    theta1, theta2 = q[4], q[3]
    if math.sin(theta2) < 0:
        return np.Inf
    # z = r2 * math.sin(theta2) - r1 * math.sin(theta1) * math.cos(theta2) - r2
    x = r2 * math.cos(theta2) + r1 * math.sin(theta1) * math.sin(theta2)
    y = y0 + r1 * math.cos(theta1)
    r = math.sqrt(x ** 2 + y ** 2)
    return r


def in_pinch(q, pinch_radius=0.105):
    """
    Return true in pinch, false otherwise
    """
    return pinchness(q) < pinch_radius


class SharedMap:
    """
    Shared memory buffer, used for communicating between processes.  Allows
    copying ints, floats, and lists of ints or floats.
    
    In particular, very useful for UR5 RTDE communication.
    """

    def __init__(self, bindings):
        """
        Create a shared memory (cross process usable) buffer.

        Parameters:
            - bindings (object or dict): A dictionary mapping from name
                str -> (class,size).
                class is either int or float, size is 0 (scalar) or
                >= 1 (array)
                Or, it can be an object containing floats, ints, or lists
                of floats or ints.  In this case, it will be compatible
                with copy_to_object / copy_from_object.
        """
        self._lock = mp.Lock()
        int_idx = 0
        float_idx = 0
        # Mapping from keynames to triples (array, start, size).
        self.names_to_starts = {}
        int_items = []
        float_items = []
        if isinstance(bindings,dict):
            for k, v in bindings.items():
                class_, size_ = v
                if size_ < 0 or not isinstance(size_, int):
                    print("Bad size {} for name {}, skipping".format(size_, k))
                    continue
                if class_ == float:
                    float_items.append((k, float_idx, size_))
                    float_idx += size_ if size_ else 1
                elif class_ == int:
                    int_items.append((k, int_idx, size_))
                    int_idx += size_ if size_ else 1
                else:
                    print("Unrecognized type {} for name {}, skipping".format(str(class_), k))
                    continue
        else:
            for k,v in bindings.__dict__.items():
                if isinstance(v,float):
                    float_items.append((k, float_idx, 0))
                    float_idx += 1
                elif isinstance(v,int):
                    int_items.append((k, int_idx, 0))
                    int_idx += 1
                elif isinstance(v,list) and len(v) > 0:
                    if isinstance(v[0],float):
                        float_items.append((k, float_idx, len(v)))
                        float_idx += len(v)
                    elif isinstance(v[0],int):
                        int_items.append((k, int_idx, len(v)))
                        int_idx += len(v)
                
        self.ints = mp.Array(ctypes.c_int, int_idx, lock=False)
        self.floats = mp.Array(ctypes.c_double, float_idx, lock=False)

        for k, idx, size in int_items:
            self.names_to_starts[k] = (self.ints, idx, size)
        for k, idx, size in float_items:
            self.names_to_starts[k] = (self.floats, idx, size)

        if not isinstance(bindings,dict):
            self.copy_from_object(bindings)

    def lock(self):
        """
        Request for the lock for this shared memory. Blocks until lock is acquired.

        A normal lock is used internally (not RLock), only call once before unlock.
        """
        self._lock.acquire()

    def unlock(self):
        """
        Release lock for this shared memory. Do not call before calling lock()
        else risk weird behavior where you release a lock being held by another thread/process.
        """
        self._lock.release()

    def __enter__(self):
        """Use with `with` blocks.

        Google terms: python context manager
        https://docs.python.org/3/reference/datamodel.html#object.__enter__
        """
        self.lock()

    def __exit__(self, exc_type, exc_val, traceback):
        self.unlock()
    
    def keys(self):
        return self.names_to_starts.keys()

    def get(self, key):
        """
        Get an entry in this shared memory.

        Parameters:
            - key:  name of entry to get.
        Return:
            Single value in the case of size-1 entries, list of values otherwise.
        """
        dat, start, size = self.names_to_starts[key]
        if size == 0:
            return dat[start]
        return dat[start : start + size]

    def get_all(self):
        """
        Get all entries in this shared memory, in a map.
        
        The returned dict is a complete mapping (key, value)
        """
        return { k: self.get(k) for k in self.keys() }

    def set(self, key, val):
        """
        Set an entry in this shared memory.
        Cannot create new entries. Entries must be created when the buffer is initialized

        Parameters:
            - key:  name of entry to set.
            - val:  int, double for size-1 entries. Iterable for larger entries.
                    NOTE: No bounds checks or type checks are performed.
        Return:
            None
        """
        dat, start, size = self.names_to_starts[key]
        if size == 0:
            dat[start] = val
        else:
            dat[start : start + size] = val

    def copy_from_object(self,obj):
        """
        Copy attributes from obj into this SharedMap assuming each key corresponds
        to an attribute name.
        """
        for k, v in self.names_to_starts.items():
            if k in obj.__dict__:
                dat, start, size = v
                if size == 0:
                    dat[start] = obj.__dict__[k]
                else:
                    dat[start:start+size] = obj.__dict__[k]
    
    def copy_to_object(self,obj):
        """
        Copy attributes from this SharedMap to obj, assuming each key corresponds
        to an attribute name.
        """
        for k, v in self.names_to_starts.items():
            if k in obj.__dict__:
                dat, start, size = v
                if size == 0:
                    obj.__dict__[k] = dat[start]
                else:
                    obj.__dict__[k] = dat[start:start+size]

class _Test:
    def __init__(self):
        self.a = 4
        self.b = 3.4
        self.c = [1,2,3]
        self.d = [0.1]
        self.e = "hello"

def self_test():
    """Tests SharedMap"""
    sm = SharedMap(_Test())
    print(sm.keys())
    foo = _Test()
    sm.ints[0] = 1
    sm.copy_to_object(foo)
    foo.b = 5.6
    sm.copy_from_object(foo)
    print(sm.ints[:])
    print(sm.floats[:])

class GlobalCollisionHelper:
    def __init__(self, klampt_model, collider):
        self.robot_model = klampt_model
        self.collider = collider

    def check_collision(self, subrobot: SubRobotModel, q1: List[float], q2: List[float]) -> bool:
        """
        Check collision between 2 limb configurations.
        
        Parameters:
        -----------
        subrobot: Subrobot whose position is being set
        q1,q2: list of doubles, limb joint positions

        Return:
        -----------
        Collision: bool, whether there is a collision
        """
        #q1 = subrobot.tofull(q1)
        #q2 = subrobot.tofull(q2)
        ## modification for just the scoopbot
        #already at the "full" config since these are not actually subrobotmodel, just robotmodel
        q1 = q1
        q2 = q2
        return self._check_collision_linear_adaptive(self.robot_model, self.collider, q1, q2)


    def _check_collision_linear_adaptive(self, robot, collider, q1, q2):
        """ Check collision between 2 robot configurations,
        but with adaptive number of collision checks

        Parameters:
        -----------------
        robot: klampt robot model
        collider: klampt collider
        q1: a list of N doubles, N = robot total DoF
        q2: a list of N doubles

        Return:
        -----------------
        bool: True if a collision is detected, false otherwise.
        """
        collision_check_interval = 0.1
        discretization = math.ceil(vo.distance(q1,q2)/collision_check_interval)
        lin = np.linspace(0,1,discretization+1)
        initialConfig = robot.getConfig()
        for _, c in enumerate(lin[1:]):
            test_config = vo.interpolate(q1, q2, c)
            robot.setConfig(test_config)
            collisions = collider.robotSelfCollisions(robot)
            for link1, link2 in collisions:
                robot.setConfig(initialConfig)
                return True
            ## add terrain collistion
            collisions = collider.robotTerrainCollisions(robot)
            for link1, link2 in collisions:
                robot.setConfig(initialConfig)
                print('Robot terrain collision')
                return True
        robot.setConfig(initialConfig)
        return False

    def check_single_q_collision(self,q):
        initialConfig = self.robot_model.getConfig()
        self.robot_model.setConfig(q)
        collisions = self.collider.robotSelfCollisions(self.robot_model)
        for link1, link2 in collisions:
            #print(link1.getName(),link2.getName())
            self.robot_model.setConfig(initialConfig)
            print('Robot self-collision')
            return True
        collisions = self.collider.robotTerrainCollisions(self.robot_model)
        for link1, link2 in collisions:
            #print(link1.getName(),link2.getName())
            self.robot_model.setConfig(initialConfig)
            print('Robot terrain collision')
            return True
        self.robot_model.setConfig(initialConfig)
        return False



class TimedLooper:
    """A class to easily control how timed loops are run.

    Usage::

        looper = TimedLooper(dt=0.01)
        while looper:
            ... do stuff ...
            if need to stop:
                looper.stop()
                #or just call break

    Note that if dt is too small (or rate is too large), the timing will not
    be accurate due to the system scheduler resolution.

    If the code within the loop takes more than dt seconds to run, then a
    warning may be printed.  To turn this off, set ``warnings=0`` in the
    constructor.  By default, this will print a warning on the first overrun,
    and every ``warning_frequency`` overruns thereafter.

    Args:
        dt (float, optional): the desired time between loops (in seconds)
        rate (float, optional): the number of times per second to run this
            loop (in Hz).  dt = 1/rate.  One of dt or rate must be specified.
        warning_frequency (int, optional): if the elapsed time between calls
            exceeds dt, a warning message will be printed at this frequency.
            Set this to 0 to disable warnings.
        name (str, optional): a descriptive name to be used in the warning
            string.

    Warning: DO NOT attempt to save some time and call the TimedLooper()
    constructor as the condition of your while loop!  I.e., do not do this::

        while TimedLooper(dt=0.01):
            ...

    """

    def __init__(self, dt=None, rate=None, warning_frequency="auto", name=None):
        self.dt = dt
        if dt is None:
            if rate is None:
                raise AttributeError("One of dt or rate must be specified")
            self.dt = 1.0 / rate
        if self.dt < 0:
            raise ValueError("dt must be positive")
        if warning_frequency == "auto":
            warning_frequency = int(2.0 / self.dt)
        self.warning_frequency = warning_frequency
        self.name = name
        self._iters = 0
        self._time_overrun_since_last_warn = 0
        self._iters_of_last_warn = 0
        self._num_overruns_since_last_warn = 0
        self._num_overruns = 0
        self._warn_count = 0
        self._tstart = None
        self._tlast = None
        self._tnext = None
        self._exit = False

    def stop(self):
        self._exit = True

    def __nonzero__(self):
        return self.__bool__()

    def __bool__(self):
        if self._exit:
            return False
        tnow = time.time()
        if self._tlast is None:
            self._tstart = tnow
            self._tnext = tnow + self.dt
        else:
            elapsed_time = tnow - self._tnext
            if elapsed_time > self.dt:
                self._num_overruns += 1
                self._num_overruns_since_last_warn += 1
                self._time_overrun_since_last_warn += elapsed_time - self.dt
                if (
                    self.warning_frequency > 0
                    and self._num_overruns % self.warning_frequency == 0
                ):
                    ave_overrun = (
                        self._time_overrun_since_last_warn
                        / self._num_overruns_since_last_warn
                    )
                    self.print_warning(
                        "{}: exceeded loop time budget {:.4f}s on {}/{} iters, by {:4f}s on average".format(
                            ("TimedLooper" if self.name is None else self.name),
                            self.dt,
                            self._num_overruns_since_last_warn,
                            self._iters - self._iters_of_last_warn,
                            ave_overrun,
                        )
                    )
                    self._iters_of_last_warn = self._iters
                    self._time_overrun_since_last_warn = 0
                    self._num_overruns_since_last_warn = 0
                    self._warn_count += 1
                self._tnext = tnow
            else:
                self._tnext += self.dt
                assert (
                    self._tnext >= tnow
                ), "Uh... elapsed time is > dt but tnext < tnow: %f, %f, %f" % (
                    elapsed_time,
                    self._tnext,
                    tnow,
                )
        self._iters += 1
        time.sleep(self._tnext - tnow)
        self._tlast = time.time()
        return True

    def time_elapsed(self):
        """Returns the total time elapsed from the start, in seconds"""
        return time.time() - self._tstart if self._tstart is not None else 0

    def iters(self):
        """Returns the total number of iters run"""
        return self._iters

    def print_warning(self, s):
        """Override this to change how warning strings are printed, e.g. to
        add your own logger"""
        print(s)
