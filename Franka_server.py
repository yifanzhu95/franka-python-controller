from xmlrpc.server import SimpleXMLRPCServer
from franka_python_controller import FrankaController
from franka_python_controller.motionUtils import GlobalCollisionHelper
from klampt import RobotModel, vis
from klampt import WorldModel,RobotModel
from klampt.model import ik,collide
from klampt.math import so3, se3, vectorops as vo

ip_address = '172.16.0.1'
port = 8080

server = SimpleXMLRPCServer((ip_address,port), logRequests=False, allow_none=True)
print(f"Listening on port {port}...")
server.register_introspection_functions()

def xmlrpcMethod(name):
    """
    Decorator that registers a function to the xmlrpc server under the given name.
    """
    def register_wrapper(f):
        server.register_function(f, name)
        return f
    return register_wrapper

## Franka Driver
world_fn = "./models/franka_world.xml"
EE_link = 'tool_link'
world = WorldModel()
world.readFile(world_fn)
robot_model = world.robot(0)
collider = collide.WorldCollider(world)
collision_checker = GlobalCollisionHelper(robot_model, collider)
params = {'address': "172.16.0.2",
        'impedance': [3000,3000,3000,2500,2500,2000,2000], #[15000, 15000, 15000, 12500, 12500, 10000, 0000],
        'payload': 1,
        'gravity_center': [0.05, 0, 0]}
        #[3000, 3000, 3000, 2500, 2500, 2000, 2000] ## default

# controller instance
global controller
controller = FrankaController(name = 'Franka', robot_model = robot_model, EE_link = EE_link, \
    collision_checker = collision_checker, params = params)

# control_params = {'kp':[600.0]*4+ [320.]*3, \
#                   'kd':[50.0]*4 + [12.5,5.0,5.]}

control_params = {'kp':[1000.0]*4+ [500.]*3, \
                  'kd':[50.0]*4 + [12.5,5.0,5.]}

# Default
# control_params = {'kp':[75.0]*5+ [50., 40.], \
#                   'kd':[25.0]*5 + [12.5]*2}


# Franka interface
@xmlrpcMethod("initialize")
def initialize():
    global controller
    controller.initialize()

@xmlrpcMethod("start")
def start():
    global controller
    controller.start()

@xmlrpcMethod("shutdown")
def shutdown():
    global controller
    controller.close()

@xmlrpcMethod("get_joint_config")
def get_joint_config():
    global controller
    return controller.get_joint_config()

@xmlrpcMethod("get_joint_velocity")
def get_joint_velocity():
    global controller
    return controller.get_joint_velocity()

@xmlrpcMethod("get_joint_torques")
def get_joint_torques():
    global controller
    return controller.get_joint_torques()

@xmlrpcMethod("get_EE_transform")
def get_EE_transform(tool_center):
    global controller
    return controller.get_EE_transform(tool_center)

@xmlrpcMethod("get_EE_velocity")
def get_EE_velocity():
    global controller
    return controller.get_EE_velocity().tolist()

@xmlrpcMethod("get_EE_wrench")
def get_EE_wrench():
    global controller
    return controller.get_EE_wrench()

@xmlrpcMethod("set_joint_config")
def set_joint_config(q):
    global controller, control_params
    controller.set_joint_config(q, control_params)

@xmlrpcMethod("set_EE_transform")
def set_EE_transform(T):
    global controller, control_params
    controller.set_EE_transform(T, control_params)

@xmlrpcMethod("set_EE_velocity")
def set_EE_velocity(v):
    global controller, control_params
    controller.set_EE_velocity(v, control_params)
    

print('Server Created')
##run server
server.serve_forever()