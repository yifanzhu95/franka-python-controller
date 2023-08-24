from klampt import RobotModel

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

def bring_in_limits(q, min_limits, max_limits):
    """
    TODO: Do this on a different layer so that we can do collision checking properly...
    """
    return np.minimum(np.maximum(q, min_limits), max_limits)

def clamp_limits(limb_lower_limit, limb_upper_limit, limb_velocity_limit,
    hw_limb_lower_limit, hw_limb_upper_limit, hw_limb_velocity_min,
    hw_lim_velocity_max):
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


class GlobalCollisionHelper:
    def __init__(self, klampt_model, collider):
        self.robot_model = klampt_model
        self.collider = collider

    def check_collision(self, subrobot, q1, q2) -> bool:
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
        q1 = subrobot.tofull(q1)
        q2 = subrobot.tofull(q2)
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
        robot.setConfig(initialConfig)
        return False
    