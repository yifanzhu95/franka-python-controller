from franka-python-controller import FrankaController
from klampt import RobotModel, vis
from klampt import WorldModel,RobotModel
from klampt.model import ik,collide
from klampt.math import so3, se3, vectorops as vo

# Klampt library for kinematics 
world_fn = "models/franka_world.xml"
EE_link = 'tool_link'
world = WorldModel()
world.readFile(world_fn)
robot_model = world.robot(0)
collider = collide.WorldCollider(world)
collision_checker = GlobalCollisionHelper(robot_model, collider)
params = {'address': ''} ## TBD, joint stiffness can also be set here 

# controller instance
controller = FrankaController(name = 'Franka', robot_model = robot_model, EE_link = EE_link, \
    collision_checker = collision_checker, params)

# start robot
controller.initialize()
controller.start() 

# get current states
current_config = controller.get_joint_config()
current_joint_velocity = controller.get_joint_velocity()
current_joint_torques = controller.get_joint_torques()

current_EE_transform = controller.get_EE_transform(tool_center = se3.identity()) #transform of tool center, 
    #specified in the tool frame (last frame of the robot)
current_EE_velocity = controller.get_EE_velocity()
current_EE_wrench = controller.get_EE_wrench() # This is in the world frame but can be easily transformed with
    # the current_EE_transform



# control the robot
controller.set_joint_config(current_config)
time.sleep(1)
controller.set_EE_transform(current_EE_transform)
time.sleep(1)
controller.set_EE_velocity([0]*6)
time.sleep(1)

# shutdown the robot
controller.close()


