from franka_python_controller import FrankaController
from franka_python_controller.motionUtils import GlobalCollisionHelper
from klampt import RobotModel, vis
from klampt import WorldModel,RobotModel
from klampt.model import ik,collide
from klampt.math import so3, se3, vectorops as vo
from icecream import ic
import copy, math, time

# Klampt library for kinematics 
world_fn = "./models/franka_world.xml"
EE_link = 'tool_link'
world = WorldModel()
world.readFile(world_fn)
robot_model = world.robot(0)
collider = collide.WorldCollider(world)
collision_checker = GlobalCollisionHelper(robot_model, collider)
params = {'address': "172.16.0.2"} ## TBD, joint stiffness can also be set here 

# controller instance
controller = FrankaController(name = 'Franka', robot_model = robot_model, EE_link = EE_link, \
    collision_checker = collision_checker, params = params)

# start robot
controller.initialize()
controller.start() 

# get current states
current_config = controller.get_joint_config()
current_joint_velocity = controller.get_joint_velocity()
current_joint_torques = controller.get_joint_torques()
ic(current_config, current_joint_torques, current_joint_velocity)

current_EE_transform = controller.get_EE_transform(tool_center = se3.identity()) #transform of tool center, 
    #specified in the tool frame (last frame of the robot)
    #also this is in the klampt se3 format, (R, t), where R is the column major format of the 3x3 rot matrix
    #and t is the translation 
current_EE_velocity = controller.get_EE_velocity()
current_EE_wrench = controller.get_EE_wrench() # This is in the world frame but can be easily transformed with
    # the current_EE_transform

ic(current_EE_transform, current_EE_velocity, current_EE_wrench)


# control the robot

# send a sinw wave motion to joint 5
control_params = {} #can alternatively specify different params
# current_time = 0.
# while current_time < 3:
#     target_q = copy.copy(current_config)
#     target_q[4] += math.sin(current_time)*0.1
#     controller.set_joint_config(target_q, control_params)
#     time.sleep(0.01)
#     current_time += 0.01
# time.sleep(1)

# # Set EE transform, in the world frame
# current_EE_transform = controller.get_EE_transform(tool_center = se3.identity()) #transform of tool center, 
# current_EE_transform[1][2] += 0.02 #move up 2 cm
# controller.set_EE_transform(current_EE_transform, control_params)
# time.sleep(1)

# Set EE velocity, in the world frame
controller.set_EE_velocity([0,0,-0.01,0,0,0], control_params)
time.sleep(5)
controller.set_EE_velocity([0,0,0,0.05,0,0], control_params)
time.sleep(5)

# shutdown the robot
controller.close()


