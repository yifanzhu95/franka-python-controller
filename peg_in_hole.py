from franka_python_controller import FrankaController
from franka_python_controller.motionUtils import GlobalCollisionHelper
from klampt import RobotModel, vis
from klampt import WorldModel,RobotModel
from klampt.model import ik,collide
from klampt.math import so3, se3, vectorops as vo
from icecream import ic
import copy, math, time
import json
import pickle
from pprint import pprint
import numpy as np
from scipy.spatial import KDTree
import PyKDL
import random


class PegInHoleTask():

    ## TODO:
    ## - Detect changes in contact formation: monitoring forces AND torque direction changes?
    ## - Trigger full contact analysis when change is detected.
    ## - Record pose when performing analysis to allow navigating from one point to another later
    ## Implement direction switching within an exploration plan linked to the evolution of the available DOFs

    def __init__(self, controller):
        
        self.controller = controller
        ## Always keep to TRUE: none of the recent tasks have been tested with the world space.
        # self.project_all_in_effector_space = False
        self.project_all_in_effector_space = True

        self.debug = False
        # self.debug = True

        self.record_forces = False
        self.recorded_forces = []
        self.record_every = 100
        self.record_file = "/data_experiments/forces/1.pkl"
        self.record_file_CF = "/data_experiments/forces/vels_FTs.pkl"
        self.recorded_delta_z = []

        # Determines how to compute the initial offset (assuming a free-floating start position)
        self.averages = [0, 0, 0, 0, 0, 0]
        self.nb_samples = 100
        self.last_forces = []
        self.delta_from_floating = [0, 0, 0, 0, 0, 0]

        # self.inHandType = rospy.Publisher('/hand_controller/movementStyle', String)
        # self.testCallback = rospy.Subscriber('/test_callback', String, self.testCallback)
        # self.testCallback2 = rospy.Subscriber('/test_callback2', String, self.testCallback2)
        # self.testCallback3 = rospy.Subscriber('/open_done', String, self.openDone)
        
        
        # self._initializeArm()
        # pprint(self._getPos())
        # self.inHandType.publish("stop")
        # self._startPos(blocking=True, idx=0) #there are three of these set up
        
        print("Arm initialized and positioned")

        # self._robot_vel_publisher = rospy.Publisher('/wam/jnt_vel_cmd', RTJointVel, queue_size=1)
        # self._robot_vel_publisher_jog = rospy.Publisher('/jog_frame', JogFrame, queue_size=1)
        
        # self._robot_joint_subscriber = rospy.Subscriber('/wam/joint_states', JointState, self.joint_state_callback )
        # self._robot_joint_state = [0, 0, 0 ,0, 0, 0 ,0]
        # # self.sensorForceSubscriber = rospy.Subscriber('/ft_sensor/ft_compensated', WrenchStamped, self.force_callback)
        # self.sensorForceSubscriber = rospy.Subscriber('/ft_compensated', WrenchStamped, self.force_callback) # New KDTree based version
        # rospy.wait_for_message('/wam/joint_states', JointState, timeout=5)
        self.force_data = None
        # self.listener = tf.TransformListener()
        
        # self.listener.waitForTransform('wam/base_link', 'sensor_space', rospy.Time(0), rospy.Duration(4))
        self.rate = 20.
        # self.r = rospy.Rate(self.rate)
        self.t = 0
        self.last_contact_time = 99
        # move = False
        self.move = True
        # self.move = False
        
        # self.current_pressed_keys = set()
        # listener = keyboard.Listener( on_press=self.on_press,     on_release=self.on_release)
        # listener.start()
        
        # Gathering the current force values, coputing the average and considering that as initial offset
        while(len(self.last_forces) < self.nb_samples):
            self.read_forces()
            time.sleep(0.05)
        
        for force in self.last_forces[:self.nb_samples-1]:
            for i in range(len(self.averages)):
                self.averages[i] += force[i]
        for i in range(len(self.averages)):
            self.averages[i] /= self.nb_samples

        
        
        print("Found initial floating z:", self.averages)

        self.read_forces()
        print("Forces after calibration:")
        pprint(self.delta_from_floating)

        # raise Exception()
        
        # Initial regulation configuration
        # self.maintain_contact = False
        self.maintain_contact = True
        self.standard_gravity_target =(None, None, -2)
        self.maintain_override = True
        self.current_contact_target = self.standard_gravity_target
        
        
        ## Defining the starting task
        self.current_task = {"name":"still", "args":{}}
        self.current_task = {"name":"slide", "args":{"slide_axis":"x"}}
        # self.current_task = {"name":"circle", "args":{}}
        # self.current_task = {"name":"orient_towards", "args":{}}
        # self.current_task = {"name":"slide_and_orient", "args":{"slide_axis":"-x"}}
        # self.current_task = {"name":"left", "args":{}}
        # self.current_task = {"name":"slide_and_climb", "args":{"slide_axis":"-x"}}
        # self.current_task = {"name":"test_contact", "args":{"frozen_axis":"y", "initial_contact_target":[None, None, None]}}
        # ONGOING: issue with the use of self.task_state and state.task_timer
        # self.current_task = {"name":"slide_and_test", "args":{"slide_axis":"x"}}
        # TO DEBUG
        # self.current_task = {"name":"align_with_plane_below", "args":{}}
        # self.current_task = {"name":"explore_hole", "args":{"start_direction":"y"}}
        # self.current_task = {"name":"rotate_against_edge", "args":{}}
        
        #TODO: @Andy these are yours
        # self.current_task = {"name":"insert_object", "args":{}}
        # self.current_task = {"name":"explore_hole_andy", "args":{}}
        
        
        # self.current_task = {"name":"insert", "args":{}}
        # self.current_task = {"name":"direct_control", "args":{"safe":True}}


        # Initializing the task "memory" variables
        self.task_state = None
        self.task_timer = time.time()
        self.task_vars = {"i":0}
        self.task_finished = False
        
        self.tasks_data = {}
        self.contact_formation_data = []
        self.current_vels = [0, 0, 0, 0, 0, 0]
        self.current_delta_forces = [0, 0, 0, 0, 0, 0]
        
        # Starting main loop
        while True:
            self.read_forces()
            print("*"*10)
            print("Delta from floating:")
            pprint(self.delta_from_floating) # Current forces with initial offset subtracted
            
            # Max speeds definition, tied to the current velocities publishing rate!!
            speed_cap = 0.2
            speed_cap_z = 0.2
            speed_cap_rot = 0.001*0.4
            speed_cap_rot_z = 0.001*0.4*3
            gains = [speed_cap, speed_cap, speed_cap_z, speed_cap_rot, speed_cap_rot, speed_cap_rot_z]
            target_motion_vels = [0, 0, 0, 0, 0, 0]
            
            ## Gathering the velocities from the regulation and main task
            # Collect the velocities from the regulation task, if applicable
            if self.maintain_contact:
                contact_vels = self.maintain_side_contact(target=self.current_contact_target)
            else:
                contact_vels = [None, None, None]
            
            # Collect the velocities from the main task
            target_motion_vels = self.get_target_motion(self.current_task["name"], self.current_task["args"], gains=gains)
            
            # If 'override' is active, erase the main task velocities.
            if not self.maintain_contact or not self.maintain_override or (self.maintain_override and self.last_contact_time < 1):
                pass
            else:
                target_motion_vels = [0, 0, 0, 0, 0, 0] # Cancelling task velocities
                print("OVERRIDE - Seeking contact!")
                # Skipping task until contact has been made.
            
            # Assembling velocites : every 'not None' velocity component from the regulation overwrites the associated component from the main task
            # Consequence: regulation command has priority over the main task.
            vels = copy.copy(target_motion_vels)
            if self.maintain_contact:
                for i, contact_vel in enumerate(contact_vels):
                    if self.current_contact_target[i] is not None and contact_vel is not None:
                        vels[i] = contact_vel
                
            ## Post-processing the velocities
            # Scaling the velocities to ensure no component is higher than its limit (if so: the entire vector is reduced iteratively until all limits are enforced)
            final_vels = self.scale_vels(vels, cap=speed_cap, cap_z=speed_cap_z, cap_rot=speed_cap_rot, cap_rot_z=speed_cap_rot_z)
            
            self.current_vels = copy.copy(final_vels)
            # Projecting the velocities from world frame to effector frame
            # (t, R)  = self.listener.lookupTransform('wam/wrist_palm_stump_link', 'wam/base_link', rospy.Time(0))
            (t, R) = self.get_pos()
            
            self.contact_formation_data.append((self.current_vels, self.delta_from_floating, time.time(), (t, R)))
            print("Velocities:")
            pprint(self.current_vels)
            print("Forces:")
            pprint(self.force_data)
            print("Delta forces")
            pprint(self.delta_from_floating)
            if self.record_forces:
                if len(self.contact_formation_data) % self.record_every == 0:
                    pickle.dump(self.contact_formation_data, open(self.record_file_CF, "wb"))
            if self.project_all_in_effector_space:
                final_vels = self.project_velocities_to_effector_space(final_vels)

            # If a force is already high in a given direction, prevent velocities that will increase it further
            final_vels = self.cancel_velocities_already_in_opposition(final_vels)

            
            if self.debug:
                print("-"*10)
                # print("Forces:", self.force_data - [self.average_x, self.average_y, self.average_z])
                print("contact speed:", contact_vels)
                print("task_speeds:", target_motion_vels)
                print("Final vels:", final_vels)
            
            # Emergency stop: if a force is over a max threshold, stop the program!
            # If ok, publishing the velocities to the controller
            max_forces = 8
            if self.move and self.are_forces_safe(self.force_data, max_forces=max_forces):    
                # self.publishJointVelocity_jog(final_vels)
                self.controller.set_EE_velocity(final_vels, {})
            else:
                if not self.are_forces_safe(self.force_data, max_forces=max_forces):
                    print("Forces are unsafe, stopping motion!")
                    return
            # self.r.sleep()
            time.sleep(1/self.rate)



    def reset_contact_estimation(self):
        print("Resetting history data for CF analysis")
        self.contact_formation_data = []



    def detect_resistance(self, ignore=[False, False, False, False, False, False]):
        """
        Check all current forces and flag those that are high enough as a sign of mechanical constraint
        """
        threshold_F = 0.5
        threshold_T = 0.05
        thresholds = [
                    threshold_F,
                    threshold_F,
                    threshold_F,
                    threshold_T,
                    threshold_T,
                    threshold_T
                    ]
        found_resistance = [[False, False], [False, False], [False, False], [False, False], [False, False], [False, False]]
        # print(found_resistance)
        for i in range(len(self.delta_from_floating)):
            if not ignore[i]:
                if self.delta_from_floating[i] < -thresholds[i]:
                    found_resistance[i][0] = True
                if self.delta_from_floating[i] > thresholds[i]:
                    found_resistance[i][1] = True
        # print("resistance result:", found_resistance)
        return found_resistance


    def _getPos(self):
        return self.arm_group.get_current_joint_values()

    def _movePos(self, pos, blocking=True):
        self._commandArmJointPos(pos, blocking=blocking)
        
    def _startPos(self, blocking=False , idx=0):
        if idx==0:
            self.arm_start_config = [0.1,0.6,0.015,1.837,0.039,0.58,1.55]
            
        elif idx==1:
            self.arm_start_config = [
                                        0.43159645167725286,
                                        0.7899865307098849,
                                        -0.4105545797555427,
                                        1.894466273038767,
                                        -0.24140429615540532,
                                        0.5600611299275257,
                                        1.8786897994969158
                                    ]
        elif idx == 2:
            self.arm_start_config = [0.10022007814186187,
                                    0.7022916998272191,
                                    0.014732710727812497,
                                    1.981050966399423,
                                    0.03811230617324119,
                                    0.3909278873869387,
                                    1.5471321302064291]
                                    
                                    
        elif idx == 3: #Andy's favorite position
            self.arm_start_config = [0.10661166475805205,
                                     0.7012599959344819,
                                     0.022030647930444076,
                                     1.8573098495099813,
                                     0.059619665673493485,
                                     0.5455911049696354,
                                     1.5538105462287042]
        
        elif idx == 4: #Andy's new favorite position
            self.arm_start_config = [0.09780953690375586, 
                                     0.5648850313760823, 
                                     0.02298850218828947, 
                                     1.7392785499976695, 
                                     0.04633570833510236, 
                                     0.7954560168108017, 
                                     1.563571308107414]
                     
                                    
        else: # Default position
            self.arm_start_config = [0.1,0.6,0.015,1.837,0.039,0.58,1.55]
        self._commandArmJointPos(self.arm_start_config, blocking=blocking)

    def _commandArmJointPos (self, pos, blocking=False):
        self.arm_group.set_joint_value_target(pos)
        plan = self._plan_execution()
        self.arm_group.execute(plan, blocking)

    def _plan_execution(self): #this keeps track of a timer
        tic = time.time()
        plan = self.arm_group.plan()
        d = time.time()-tic
        self.planning_time+=d
        self.planning_actions+=1
        return plan

    def reset_counters(self):
        self.planning_time = 0.
        self.planning_actions = 0
        self.start_time = time.time()
        self.hand_actions = 0

    # def _initializeArm(self):
    #     self.reset_counters()
    #     self.arm_group = MoveGroupCommander("arm")
    #     self._pos_tolerance = 0.0
    #     self._ortn_tolerance = 0.0
    #     self.planner_type = 'RRTConnectkConfigDefault'
    #     self.arm_group.set_planner_id(self.planner_type)

    def project_velocities_to_effector_space(self, vels):
        test = True
        
        if not test:
            # From world to effector
            (trans1, rot1)  = self.listener.lookupTransform('wam/base_link', 'wam/wrist_palm_stump_link', rospy.Time(0))
            rotation = Rotation.from_quat(rot1)
            rotation_debug = Rotation.from_euler("z",0, degrees=True )
            
            debug = False
            if debug:
                rotation = rotation_debug
            # print("Applying rotation:", rotation.as_euler("xyz", degrees=True))
            T = [vels[0], vels[1], vels[2]]
            R = [vels[3], vels[4], vels[5]]
            
            # Something is wrong, a 90deg rotation around the z axis seem to solve the issue. Should be looked into at some point, probably a dump mistake.
            rot_T = rotation_debug.apply(rotation.apply(T))
            rot_R = rotation_debug.apply(rotation.apply(R))

            # print("Before rotation:", vels)
            # print("After rotation:", rot_T, rot_R)

            return [rot_T[0], rot_T[1], rot_T[2], rot_R[0], rot_R[1], rot_R[2]]
        else:
            return [vels[0], vels[1], -vels[2], vels[3], vels[4], vels[5]]
    # def rotate_force_readings(self, forces):
    #     try:
    #         if not self.project_all_in_effector_space:
    #             # From sensor to world
    #             (trans1, rot1)  = self.listener.lookupTransform('wam/base_link', 'sensor_space', rospy.Time(0))
    #         else:
    #             # From sensor to effector
    #             rotation_debug = Rotation.from_euler("z",0, degrees=True )
    #             (trans1, rot1)  = self.listener.lookupTransform('wam/wrist_palm_stump_link', 'sensor_space', rospy.Time(0))
    #         rotation = Rotation.from_quat(rot1)
    #         # print("Applying rotation:", rotation.as_euler("xyz", degrees=True))
    #         forces = self.force_data[0:3]
    #         torques = self.force_data[3:6]
    #         if self.project_all_in_effector_space:
    #             aligned_forces = rotation_debug.apply(rotation.apply(forces))
    #             aligned_torques = rotation_debug.apply(rotation.apply(torques))
    #         else:
    #             aligned_forces = rotation.apply(forces)
    #             aligned_torques = rotation.apply(torques)
            
    #         # print("Initial forces:", forces)
    #         # print("Rotated forces:", aligned_forces)
    #         # print("Rotated torques:", aligned_forces)
    #         # To avoid breaking the contact regulation code when switching back and forth from world to effector
    #         # aligned_forces[2]*=-1
            
    #         result = [
    #             aligned_forces[0],
    #             aligned_forces[1],
    #             aligned_forces[2],
    #             aligned_torques[0],
    #             aligned_torques[1],
    #             aligned_torques[2],
    #             ]

    #         return result
    #     except Exception as e:
    #         print("Rotation failed:", e)
    #         return forces
    
    def cancel_velocities_already_in_opposition(self, vels):
        forces_safety_threshold = 2
        for i, vel in enumerate(vels):
            if vel < 0 and self.delta_from_floating[i] < -forces_safety_threshold:
                print("Cancelling velocity:", -(i+1))
                vels[i] = 0
            if vel > 0 and self.delta_from_floating[i] > forces_safety_threshold:
                print("Cancelling velocity:", i+1)
                vels[i] = 0
        return vels

    def are_forces_safe(self, forces, max_forces=10, max_torques=1):
        # pprint(forces)
        return abs(forces[0]) < max_forces and abs(forces[1]) < max_forces and abs(forces[2]) < max_forces and abs(forces[3]) < max_torques and abs(forces[4]) < max_torques and abs(forces[5]) < max_torques
    
    def scale_vels(self, vels, cap, cap_z, cap_rot, cap_rot_z):
        final_vels = vels
        while(abs(final_vels[0]) > cap or abs(final_vels[1]) > cap):
            # print("Speeds too high, scaling down...")
            final_vels[0] *= 0.9
            final_vels[1] *= 0.9
            
        while(abs(final_vels[2])) > cap_z:
            final_vels[2] *= 0.9

        while(abs(final_vels[3]) > cap_rot or abs(final_vels[4]) > cap_rot):
            final_vels[3] *=0.9
            final_vels[4] *=0.9
            
        while abs(final_vels[5]) > cap_rot_z:
            final_vels[5] *=0.9

        return final_vels
    
    def get_target_motion(self, task_type="still", args=None, gains=[0.001, 0.001, 0.001, 0, 0, 0]):
        """
        Collect the velocities associated to a given main task
        Velocities are [tx, ty, tz, rx, ry, rz] (for the lateral axis orientation, refer to the markings on the robot hand)
        """
        print("Executing task:", task_type)
        vels = [0, 0, 0, 0, 0, 0]
        if task_type == "still":
            vels = [0, 0, 0, 0, 0, 0]
        if task_type == "down":
            vels = [0, 0, 1, 0, 0, 0]
        if task_type == "up":
            vels = [0, 0, -1, 0, 0, 0]
        if task_type == "left":
            vels = [-1, 0, 0, 0, 0, 0]
        if task_type == "right":
            vels = [1, 0, 0, 0, 0, 0]
        if task_type == "forward":
            vels = [0, 1, 0, 0, 0, 0]
        if task_type == "backward":
            vels = [0, -1, 0, 0, 0, 0]
        if task_type == "circle":
            vels = self.draw_cirle(axis=args["axis"])
        if task_type == "orient_towards":
            # print("Orienting")
            vels = self.orient_towards()
        if task_type == "slide":
            vels = self.slide(axis=args["slide_axis"])
        if task_type == "slide_and_orient":
            if abs(self.force_data[0]-self.average_x) > 1 or abs(self.force_data[1]-self.average_y) > 1:
                vels = self.orient_towards()
            else:
                print("Sliding!")
                if args["slide_axis"] == "x":
                    vels = [1, 0, 0, 0, 0, 0]
                if args["slide_axis"] == "-x":
                    vels = [-1, 0, 0, 0, 0, 0]
                if args["slide_axis"] == "y":
                    vels = [0, 1, 0, 0, 0, 0]
                if args["slide_axis"] == "-y":
                    vels = [0, -1, 0, 0, 0, 0]
        if task_type == "slide_and_climb":
            vels = self.explore_laterally(axis=args["slide_axis"]) 
        if task_type == "test_contact":
            vels = self.test_contact(frozen_axis=args["frozen_axis"], initial_contact_target=args["initial_contact_target"])
        if task_type == "align_with_plane_below":
            vels = self.align_with_plane_below()
        if task_type == "slide_and_test":
            vels = self.slide_and_test(args["slide_axis"])
        if task_type == "direct_control":
            vels = self.direct_control()
        if task_type == "explore_hole":
            vels = self.explore_hole(args["start_direction"])
        if task_type == "rotate_against_edge":
            vels = self.rotate_against_edge()
        if task_type == "insert_object":
            vels = self.insert_object()
        if task_type =="explore_hole_andy":
            vels = self.explore_hole_andy()

        return self.apply_gains(vels, gains)
        
    def pick_exploration_direction(self):
        # IPython.embed()
        KD = KDTree(self.task_vars['positions'])
        candidates = []
        for _ in range(40):
            candidates.append([np.random.uniform(self.task_vars["workspace_bounds"][0][0],self.task_vars["workspace_bounds"][0][1]),np.random.uniform(self.task_vars["workspace_bounds"][1][0],self.task_vars["workspace_bounds"][1][1])])

        max_idx= 0
        for i in range(1,len(candidates)):
            if KD.query(candidates[i])[0] > KD.query(candidates[max_idx])[0]:
                max_idx = i 
        
        return candidates[max_idx]
    
    def explore_hole_andy(self):
        
        '''
        Task: 
        - (1) Explore: Go in a given lateral direction
            - If edge of workspace is reached, pick a new direction and start over at (1)
            - If a contact is detected, switch to (2)
        - (2) Insert: Regulate to center the object, push down and spiral the fingers
        '''

        vels = [0, 0, 0, 0, 0, 0]

        keep_within_workspace = True

        inhand_manipulation = True

        # (trans1, rot1)  = self.listener.lookupTransform('wam/wrist_palm_stump_link', 'wam/base_link', rospy.Time(0))
        (trans1, rot1) = self.get_pos()

        file_path = "/home/grablab/Documents/data_experiments/forces/hole_exploration.pkl"
        
        z_target = -0.8 #this was 0.5
        
        if "is_started" not in self.task_vars:
            if inhand_manipulation:
                self.inHandType.publish("stop") #reset the grasp
            


            self.current_contact_target = (-0.5, -0.5, z_target)

            self.task_vars.update({"is_started":True})
            self.task_vars.update({"contact_started":False})
            self.maintain_override = True
            self.task_vars.update({"recorded":[]})
            self.task_vars.update({"positions":[]})
            self.task_vars.update({"global_timer":time.time()})
            self.task_vars['target'] = [0,0]
                    
            #Calculate workspace
            center_point = [trans1[0], trans1[1]]
            workspace_size = 0.01 # cm
            self.task_vars.update({"workspace_bounds":[[center_point[0]-workspace_size/2, center_point[0]+workspace_size/2], [center_point[1]-workspace_size/2, center_point[1]+workspace_size/2]]})
            self.task_vars['target'] = copy.copy(center_point)
            
            #self.task_vars.update({"task":"explore"})
            self.task_vars.update({"task":"initial"})
            pickle.dump([[], self.task_vars["workspace_bounds"], [], self.task_vars['target']], open(file_path, "wb"))
            self.task_vars.update({"global_timer":time.time()})
            self.task_vars.update({'recently_inside_workspace':True})
            self.task_vars.update({'contact_target_for_insertion':copy.copy(self.current_contact_target)})      
            self.task_vars.update({"time_outside_workspace": 1e12}) #large if inside workspace, timer if outside
            self.task_timer = time.time()
            time.sleep(0.5)

        print("Sub task:", self.task_vars["task"])

        if self.task_vars["task"] == "initial":
            if time.time() - self.task_timer > 5:
                self.current_contact_target = (-0.5, 0, z_target)
                self.task_vars.update({"task":"explore"})
            else:
                self.current_contact_target = [-0.5, 0,None] 
            
            new_record = self.task_vars["positions"]
            new_record.append((trans1[0], trans1[1]))
            self.task_vars.update({"positions":new_record})
            
                

        if self.task_vars["task"] == "explore":
            if keep_within_workspace:
                print("T:", trans1[0], "/", trans1[1])
                print(self.task_vars["workspace_bounds"])
                randomness = 0.25
                target_lateral = 0.8
                
                outside_workspace = (trans1[0] < self.task_vars["workspace_bounds"][0][0] and self.current_contact_target[0] > 0) or \
                                    (trans1[0] >  self.task_vars["workspace_bounds"][0][1] and self.current_contact_target[0] < 0) or \
                                    (trans1[1] < self.task_vars["workspace_bounds"][1][0] and self.current_contact_target[1] > 0) or \
                                    (trans1[1] >  self.task_vars["workspace_bounds"][1][1] and self.current_contact_target[1] < 0)
                
                if outside_workspace and self.task_vars['recently_inside_workspace'] or time.time()- self.task_vars['time_outside_workspace']> 4:
                    target = self.pick_exploration_direction()
                    
                    self.task_vars['target'] = target
                    target.append(0)
                    diff = np.asarray(trans1)-np.asarray(target)
                    if abs(diff[0])> abs(diff[1]):
                        mult = abs(0.5/diff[0])
                    else:
                        mult = abs(0.5/diff[1])
                    self.current_contact_target = [mult*diff[0], mult*diff[1], -0.7]
                    #self.current_contact_target = [0.5, 0, -0.7]
                    self.task_vars.update({'contact_target_for_insertion':copy.copy(self.current_contact_target)})                    
                    self.task_vars.update({'recently_inside_workspace':False})
                    self.task_vars['time_outside_workspace'] = time.time()
                    
                #This is so we only change once while we are outside the workspace
                if not outside_workspace:
                    self.task_vars.update({'recently_inside_workspace':True})
                    self.task_vars['time_outside_workspace'] = 1e12
                    

            # According to the current target, we set the grasp configuration to angle the object in the same direction as the motion
            if inhand_manipulation:
                if abs(self.current_contact_target[0]) > abs( self.current_contact_target[1]):
                    if self.current_contact_target[0] > 0:
                        self.inHandType.publish("left")
                    else:
                        self.inHandType.publish("right")
                else:
                    if self.current_contact_target[1] > 0:
                        self.inHandType.publish("up")
                    else:
                        self.inHandType.publish("down")
            # 
            # if not self.task_vars["contact_started"]:
            #     self.task_timer = time.time()

            if self.last_contact_time < 1 and not self.task_vars["contact_started"]:
                self.task_timer = time.time()
                self.task_vars.update({"contact_started":True})
                print("Started task timer")
            # if self.last_contact_time > 1 and self.task_vars["contact_started"]:
            #     self.task_vars.update({"contact_started":False})

            print("task timer:", time.time()-self.task_timer)
            lateral_strength = 0.5
            if np.linalg.norm(np.asarray([self.delta_from_floating[0], self.delta_from_floating[1]])) > lateral_strength:
            #if self.last_contact_time < 0.1:
                self.current_task = {"name":"insert_object", "args":{}}
            

            new_record = self.task_vars["positions"]
            new_record.append((trans1[0], trans1[1]))
            self.task_vars.update({"positions":new_record})
                
            if time.time() - self.task_vars["global_timer"] > 1:
                pickle.dump([self.task_vars["recorded"], self.task_vars["workspace_bounds"], self.task_vars["positions"],self.task_vars['target'] ], open(file_path, "wb"))
                self.task_vars.update({"global_timer":time.time()})

        return vels

    #New version of the old one, hopefully we can get it to work this time. 
    #Works pretty well for the pear, still working on for the triangle. 
    def insert_object(self):
        """
        Task:
        - (1) Go down until contact
        - (2) Go right until side contact
        - (3) Go left until side contact
        - (4) Rotate until you find get a torque that is too high
        - (5) Align and insert
        """
        
        vels = [0, 0, 0, 0, 0, 0]
        insertion_done = False
        # (t, R)  = self.listener.lookupTransform('wam/wrist_palm_stump_link', 'wam/base_link', rospy.Time(0))
        (t, R) = self.get_pos()
        lateral_strength = 0.5
        downward_strength = 0.5


        # Initializing the task. Starting with 'down' task
        if "state" not in self.task_vars:
            self.task_vars.update({"state":"down"})
            self.task_vars. update({"starting_position": t})
            self.maintain_override = False
            
        print('********************************')
        print('Task: ', 'insert_object')
        print("Sub-task:", self.task_vars["state"])
        
        # Sub-task: Go down, and get a contact | Do nothing until self.last_contact is ~ 0
        if self.task_vars["state"] == "down":
            self.current_contact_target = self.standard_gravity_target
            if abs(self.delta_from_floating[2]) > abs(self.standard_gravity_target[2]): #downward_strength: # Regulation finished, switch to slide1
                self.task_vars.update({"state":"slide1"})
            else:
                vels = [0, 0, 0, 0, 0, 0] # Do nothing, let the regulation work


        # Sub-task: Go left, keep down, until catching the edge of the hole.
        if self.task_vars["state"] == "slide1":
            # IPython.embed()
            if 'contact_target_for_insertion' in self.task_vars.keys():
                self.current_contact_target = self.task_vars['contact_target_for_insertion']
                
                unit_vector_1 =np.asarray(self.delta_from_floating[:2]) / np.linalg.norm(np.asarray(self.delta_from_floating[:2]))
                unit_vector_2 = np.asarray(self.current_contact_target[:2]) / np.linalg.norm(np.asarray(self.current_contact_target[:2]))
                dot_product = np.dot(unit_vector_1, unit_vector_2)
                angle = np.arccos(dot_product)
                print('ANGLE1: ', angle)
                print('NORM1: ', np.linalg.norm(np.asarray(self.delta_from_floating[:2])))
                if abs(angle)<0.3 and np.linalg.norm(np.asarray(self.delta_from_floating[:2]))> lateral_strength:
                    self.task_vars.update({"state":"slide2"})
                
            else:
                # self.inHandType.publish("left")
                self.current_contact_target = (lateral_strength, None, -downward_strength)
                if abs(self.delta_from_floating[0]) > lateral_strength: # Regulation finished, switch to slide1
                    self.task_vars.update({"state":"slide2"})
                else:
                    vels = [0, 0, 0, 0, 0, 0] # Do nothing, let the regulation work
                
        # Sub-task: Go up
        if self.task_vars["state"] == "slide2":
            if 'contact_target_for_insertion' in self.task_vars.keys():
                self.current_contact_target = [-self.task_vars['contact_target_for_insertion'][1]+self.task_vars['contact_target_for_insertion'][0], self.task_vars['contact_target_for_insertion'][0]+self.task_vars['contact_target_for_insertion'][1], self.task_vars['contact_target_for_insertion'][2]]
                
                
                unit_vector_1 =np.asarray(self.delta_from_floating[:2]) / np.linalg.norm(np.asarray(self.delta_from_floating[:2]))
                unit_vector_2 = np.asarray(self.current_contact_target[:2]) / np.linalg.norm(np.asarray(self.current_contact_target[:2]))
                dot_product = np.dot(unit_vector_1, unit_vector_2)
                angle = np.arccos(dot_product)
                print('ANGLE2: ', angle)
                print('NORM2: ', np.linalg.norm(np.asarray(self.delta_from_floating[:2])))
                
                if abs(angle)<0.5 and np.linalg.norm(np.asarray(self.delta_from_floating[:2]))> lateral_strength/1.5:
                    self.task_vars.update({"state":"rotate"})
                    self.past_contact_target = copy.copy(self.current_contact_target)
            
            else:
                self.current_contact_target = (lateral_strength, lateral_strength, -downward_strength)
                if abs(self.delta_from_floating[1]) > lateral_strength: # Regulation finished, switch to rotate
                    self.task_vars.update({"state":"rotate"})
                    self.past_contact_target = copy.copy(self.current_contact_target)
                    
                else:
                    vels = [0, 0, 0, 0, 0, 0] # Do nothing, let the regulation work

        # Sub-task: Go up
        if self.task_vars["state"] == "rotate":
            self.current_contact_target = [np.clip(self.past_contact_target[0], -0.4, 0.4),np.clip(self.past_contact_target[1], -0.4, 0.4), -1.1] #self.past_contact_target[2]]
            vels = [0, 0, 0, 0, 0, 1] # Do nothing, let the regulation work
            #if abs(self.delta_from_floating[5]) > 0.02:# Regulation finished, switch to slide1
            if self.delta_from_floating[5] > 0.05:# Regulation finished, switch to slide1 USED 0.01 for the triangle
                self.task_vars.update({"state":"insert"})


        if self.task_vars["state"] == "insert":
            rots = self.getObjectAngle()
            if rots[0] is not None:
                print('Rotations about axes: ', rots)
                try:
                    #Always use the hand rotations
                    if abs(rots[1])> abs(rots[0]):
                        if rots[1]<-0.0:
                            self.inHandType.publish("right_delta")
                        elif rots[1]>0.0:
                            self.inHandType.publish("left_delta")
                    else:
                        if rots[0]<-0.0:
                            self.inHandType.publish("up_delta")
                        elif rots[1]>0.0:
                            self.inHandType.publish("down_delta")
                except:
                    pass
                    # rospy.logwarn('ANDY YOU DID SOMEHTING BAD ')
                    # IPython.embed()
                    
                #regulate_motion = "translation" #this worked fine for the pear
                regulate_motion = "rotation"

                if regulate_motion == "translation":
                    tar = 1.0
                    self.current_contact_target = [0, 0, -1.0] #use this for the normal stuff
                    # self.current_contact_target = [self.current_contact_target[0], self.current_contact_target[1], -1.5]
                    
                    if rots[1]<-0.02:
                        self.current_contact_target[0] = -tar
                    elif rots[1]>0.02:
                        self.current_contact_target[0] = tar
                    if rots[0]<-0.02:
                            self.current_contact_target[1] = -tar
                    elif rots[0]>0.02:
                            self.current_contact_target[1] = tar
                elif regulate_motion == "rotation":
                    tar = 1.0
                    self.current_contact_target = [0, 0, -0.8] #use this for the normal stuff
                    if rots[1]<-0.0:
                        vels[3] = -1
                        # self.current_contact_target[0] = -tar
                    elif rots[1]>0.0:
                        vels[3] = 1
                        # self.current_contact_target[0] = tar
                    if rots[0]<-0.0:
                        vels[4] = 1
                        # self.current_contact_target[1] = -tar
                    elif rots[0]>0.0:
                        vels[4] = -1
                        # self.current_contact_target[1] = tar
            else:
                # self.inHandType.publish("spiral")
                self.current_contact_target = [0, 0, -1.]
            
        return vels

    #Original, sorta working
    def rotate_against_edge(self):
        """
        Task:
        - (1) Go down until contact
        - (2) Go right until side contact
        - (3) Start rotating aroung z - if jammed (torque on rz), pull up
        - (4) If contact is lost, get back, and restart at (3)
        - (5) If altitude get low enough, start inserting: regulate lateral forces to 0, push down, and spiral fingers
        """
        
        vels = [0, 0, 0, 0, 0, 0]
        insertion_done = False
        
        # (t, R)  = self.listener.lookupTransform('wam/wrist_palm_stump_link', 'wam/base_link', rospy.Time(0))
        (t, R) = self.get_pos()
        # (t, R)  = self.listener.lookupTransform('tag_10', 'wam/base_link', rospy.Time(0))
        print('ALTITUDE: ')
        print(t)
        print('-----------------')
        
        if "ref_altitude" in self.task_vars:
            print("Altitude delta:", self.task_vars["ref_altitude"] - t[2])
        
        # offset_altitude = 0.005 #works decently for the cylinder
        
        #offset_altitude = 0.014 # Works with pear ## OLD
        offset_altitude = 0.005 # Works with pear
        # offset_altitude = 0.01 # Works with triangle/rectangle  (not really)

        lateral_strength = 0.4

        # Initializing the task. Starting with 'down' task
        if "state" not in self.task_vars:
            self.task_vars.update({"state":"down"})
            # self.task_vars.update({"state":"insert"})
            self.task_vars.update({"initial_contact":False})
            self.maintain_override = False
            self.current_contact_target = self.standard_gravity_target

        print("Sub-task:", self.task_vars["state"])
        print("Initial contact:", self.task_vars["initial_contact"])

        # Sub-task: Go down, and get a contact | Do nothing until self.last_contact is ~ 0
        if self.task_vars["state"] == "down":
            if self.last_contact_time < 0.5 and not self.task_vars["initial_contact"]:
                # Regulation finished, switch to keep_edge. Start timer to prevent switching to 'going_back' immediately.
                self.task_vars.update({"state":"keep_edge"})
                self.task_timer = time.time()
                
            else:
                vels = [0, 0, 0, 0, 0, 0] # Do nothing, let the regulation work

        # Sub-task: Go left, keep down, until catching the edge of the hole.
        if self.task_vars["state"] == "keep_edge":
            
            self.inHandType.publish("left")
            self.current_contact_target = (lateral_strength, lateral_strength, -1)
            
            if self.task_vars["initial_contact"]:
                if "ref_altitude" not in self.task_vars:
                    if "alt_timer" not in self.task_vars:
                        self.task_vars.update({"alt_timer":time.time()})
                        self.task_vars.update({"alt_hist":[]})
                    else:
                        if time.time() - self.task_vars["alt_timer"] < 2:
                            all_alts = self.task_vars["alt_hist"]
                            all_alts.append(t[2])
                            self.task_vars.update({"alt_hist":all_alts})
                            return vels
                        else:
                            ref_alt = 0
                            for alt in self.task_vars["alt_hist"]:
                                ref_alt += alt
                            ref_alt /= len(self.task_vars["alt_hist"])
                        
                            print(ref_alt)
                            #IPython.embed()
                            self.task_vars.update({"ref_altitude":ref_alt})
                else:
                    vels[5] = 1
                #self.current_contact_target = (lateral_strength, lateral_strength , -1)
                
            
            if not self.task_vars["initial_contact"] and time.time() - self.task_timer > 5 and self.last_contact_time < 0.1:
                self.task_vars.update({"initial_contact":True})
               
            
            if self.task_vars["initial_contact"] and self.last_contact_time > 10:
                print("Edge lost, going back to retrieve it!")
                self.task_vars.update({"state":"going_back"})
                self.task_timer = time.time()
                # self.task_timer = time.time()

            if "ref_altitude" in self.task_vars and self.task_vars["ref_altitude"]-t[2] > offset_altitude \
                or abs(self.delta_from_floating[5]) > 0.06: #or self.delta_from_floating[0]>0.5:
                
                self.task_vars.update({"state":"insert"})
                self.task_timer = time.time()
                
        # Condition to stop going back is broken: stops WAY TOO SOON.
        if self.task_vars["state"] == "going_back":
            # self.inHandType.publish("right")
            self.current_contact_target = (-lateral_strength, None, -0.5)
            # if self.last_contact_time <0.1:
            if time.time() - self.task_timer > 10:
                print("Contact re-establish with other edge of hole.")
                self.task_vars.update({"state":"keep_edge"})
                self.task_vars.update({"initial_contact":False})
            vels[5] = 1

        if abs(self.delta_from_floating[5]) > 0.06:
            # Getting in a 'jammed' situation. Needs to pull out, keep rotating and enter again.
            print("JAMMED!! Pulling out...")
            self.current_contact_target = (None, None, None)
            vels = [0, 0, -1, 0, 0, 0]


        if self.task_vars["state"] == "insert":
            # IPython.embed()
            rots = self.getObjectAngle()
            print('Rotations about axes: ', rots)
            
            #Always use the hand rotations
            if abs(rots[1])> abs(rots[0]):
                if rots[1]<-0.0:
                    self.inHandType.publish("right_delta")
                elif rots[1]>0.0:
                    self.inHandType.publish("left_delta")
            else:
                if rots[0]<-0.0:
                    self.inHandType.publish("up_delta")
                elif rots[1]>0.0:
                    self.inHandType.publish("down_delta")
                
            if time.time() - self.task_timer < 5:
                # self.current_contact_target = [None, None, None]
                self.current_contact_target = [0.5, 0, -0.5]
                # self.inHandType.publish("open")
                
                #vels = [0, 0, 0, 0, 0, -1]
            #when out of limits, use the arm every 5 seconds or so. 
            else:

                # regulate_motion = "translation"
                regulate_motion = "rotation"

                if regulate_motion == "translation":
                    tar = 1.5
                    self.current_contact_target = [0, 0, -tar]
                    if rots[1]<-0.0:
                        self.current_contact_target[0] = -tar
                    elif rots[1]>0.0:
                        self.current_contact_target[0] = tar
                    if rots[0]<-0.0:
                            self.current_contact_target[1] = -tar
                    elif rots[0]>0.0:
                            self.current_contact_target[1] = tar
                elif regulate_motion == "rotation":
                    if rots[1]<-0.0:
                        vels[3] = -1
                    elif rots[1]>0.0:
                        vels[3] = 1
                    if rots[0]<-0.0:
                        vels[4] = 1
                    elif rots[0]>0.0:
                        vels[4] = -1
                # vels = [0, 0, 0, 0, 0, 1]
                # self.inHandType.publish("spiral")                
                
            if time.time() - self.task_timer > 10:
                self.task_timer = time.time()
            
            # if self.task_vars["ref_altitude"]-t[2] > 0.045:        
            #     IPython.embed()
            
        return vels


    def getObjectAngle(self):
        try:
            # (t, R)  = self.listener.lookupTransform('tag_10', 'wam/wrist_palm_stump_link', rospy.Time(0))
            (t, R) = self.get_pos()
            rot = PyKDL.Rotation.Quaternion(*R)
            rots = list(rot.GetRPY())
            rots[0]= rots[0]+math.pi
            rots[0] = (rots[0]+math.pi)%(2*math.pi) - math.pi
            return rots
        except:
            return [None, None, None]


    def explore_hole(self, start_direction):
        '''
        Task: 
        - (1) Explore: Go in a given lateral direction
            - If edge of workspace is reached, pick a new direction and start over at (1)
            - If a contact is detected, switch to (2)
        - (2) Insert: Regulate to center the object, push down and spiral the fingers
        '''
        # self.current_contact_target = [None, None, None]
        # return [0, 0, 0, 0, 0, 1]

        

        vels = [0, 0, 0, 0, 0, 0]

        keep_within_workspace = True
        # keep_within_workspace = False

        randomize_new_direction = True
        # randomize_new_direction = False

        inhand_manipulation = True

        center_point = [-0.027, -0.64]
        workspace_size = 0.01 # cm
        workspace_bounds = [[center_point[0]-workspace_size/2, center_point[0]+workspace_size/2], [center_point[1]-workspace_size/2, center_point[1]+workspace_size/2]]
        
        # (trans1, rot1)  = self.listener.lookupTransform('wam/wrist_palm_stump_link', 'wam/base_link', rospy.Time(0))
        (trans1, rot1)  = self.get_pos()
        print("Workspace bounds")
        pprint(workspace_bounds)
        print("T:", trans1)
        
        file_path = "/home/grablab/Documents/data_experiments/forces/hole_exploration.pkl"
        
        
        if "is_started" not in self.task_vars:
            if inhand_manipulation:
                self.inHandType.publish("stop")
            z_target = -0.5
            # z_target = None
            if start_direction == "-x":
                self.current_contact_target = (-0.5, 0, z_target)
            if start_direction == "x":
                self.current_contact_target = (0.5, 0, z_target)
            if start_direction == "-y":
                self.current_contact_target = (0, -0.5, z_target)
            if start_direction == "y":
                self.current_contact_target = (0, 0.5, z_target)
            if start_direction == "xy":
                self.current_contact_target = (0.5, 0.5, z_target)
            self.task_vars.update({"is_started":True})
            self.task_vars.update({"contact_started":False})
            self.maintain_override = True
            self.task_vars.update({"recorded":[]})
            self.task_vars.update({"positions":[]})
            self.task_vars.update({"global_timer":time.time()})

            self.task_vars.update({"task":"explore"})

            pickle.dump([[], workspace_bounds, []], open(file_path, "wb"))
            self.task_vars.update({"global_timer":time.time()})

        print("Sub task:", self.task_vars["task"])

        if self.task_vars["task"] == "explore":
            if keep_within_workspace:
                print("T:", trans1[0], "/", trans1[1])
                randomness = 0.25
                if trans1[0] < workspace_bounds[0][0] and self.current_contact_target[0] > 0:
                    print("-x bound reached, inverting target")
                    self.current_contact_target = (-self.current_contact_target[0]+random.uniform(-randomness,randomness), self.current_contact_target[1]+random.uniform(-randomness,randomness), self.current_contact_target[2])
                if trans1[0] > workspace_bounds[0][1] and self.current_contact_target[0] < 0:
                    print("+x bound reached, inverting target")
                    self.current_contact_target = (-self.current_contact_target[0]+random.uniform(-randomness,randomness), self.current_contact_target[1]+random.uniform(-randomness,randomness), self.current_contact_target[2])
            
                if trans1[1] < workspace_bounds[1][0] and self.current_contact_target[1] > 0:
                    print("-y bound reached, inverting target")
                    self.current_contact_target = (self.current_contact_target[0]+random.uniform(-randomness,randomness), -self.current_contact_target[1]+random.uniform(-randomness,randomness), self.current_contact_target[2])
                if trans1[1] > workspace_bounds[1][1] and self.current_contact_target[1] < 0:
                    print("+y bound reached, inverting target")
                    self.current_contact_target = (self.current_contact_target[0]+random.uniform(-randomness,randomness), -self.current_contact_target[1]+random.uniform(-randomness,randomness), self.current_contact_target[2])
            

            # According to the current target, we set the grasp configuration to angle the object in the same direction as the motion
            if inhand_manipulation:
                if abs(self.current_contact_target[0]) > abs( self.current_contact_target[1]):
                    if self.current_contact_target[0] > 0:
                        self.inHandType.publish("left")
                    else:
                        self.inHandType.publish("right")
                else:
                    if self.current_contact_target[1] > 0:
                        self.inHandType.publish("up")
                    else:
                        self.inHandType.publish("down")

            if not self.task_vars["contact_started"]:
                self.task_timer = time.time()

            if self.last_contact_time < 1 and not self.task_vars["contact_started"]:
                self.task_timer = time.time()
                self.task_vars.update({"contact_started":True})
                print("Started task timer")
            if self.last_contact_time > 1 and self.task_vars["contact_started"]:
                self.task_vars.update({"contact_started":False})

            print("task timer:", time.time()-self.task_timer)
            if self.task_vars["contact_started"] and time.time() - self.task_timer > 1:
                max_target = 0.5
                print("Equilibrium reached, switching direction")
                print("torques:", self.delta_from_floating[3], self.delta_from_floating[4])
                ratio = abs(self.delta_from_floating[3])/(abs(self.delta_from_floating[3])+abs(self.delta_from_floating[4]))
                print("Ratio:", ratio)

                # X velocity is driven by the amount of Y-axis torque, and Y vel by X-torque
                new_y = abs(self.delta_from_floating[3])/self.delta_from_floating[3] # Getting the sign of current direction
                # print("new x sign:", new_y)
                new_y *= ratio*max_target
                # print("new x:", new_y)
                
                new_x = abs(self.delta_from_floating[4])/self.delta_from_floating[4] # Getting the sign of current direction
                new_x *= -(1-ratio)*max_target
                
                new_x_target = new_x
                new_y_target = new_y

                if randomize_new_direction:
                    amount = 0.5
                    
                    new_x_target += random.uniform(-amount, amount)
                    new_y_target += random.uniform(-amount, amount)
                
                    new_x_target = np.clip(new_x_target, -max_target, max_target)
                    new_y_target = np.clip(new_y_target, -max_target, max_target)

                # new_x = None
                # self.current_contact_target = ( 
                #     -self.current_contact_target[0] if self.current_contact_target[0] is not None else None,
                #     -self.current_contact_target[1] if self.current_contact_target[1] is not None else None,
                #     self.current_contact_target[2]
                #                               )
                self.current_contact_target = (new_x_target, new_y_target, self.current_contact_target[2])
                print("new target:", self.current_contact_target)
                # time.sleep(9999999999999999999)
                self.task_timer = time.time()
                
                self.task_vars.update({"last_torque_direction":(self.delta_from_floating[3], self.delta_from_floating[4])})
                
                
                new_record = self.task_vars["recorded"]
                # new_record.append((trans1, (self.delta_from_floating[0], self.delta_from_floating[1])))
                new_record.append((trans1, (new_x, new_y)))
                self.task_vars.update({"recorded":new_record})

                # Switching to new insertion task: TESTING
                self.task_vars.update({"task":"insert"})
                self.inHandType.publish("spiral")

            new_record = self.task_vars["positions"]
            new_record.append((trans1[0], trans1[1]))
            self.task_vars.update({"positions":new_record})
                
            if time.time() - self.task_vars["global_timer"] > 1:
                pickle.dump([self.task_vars["recorded"], workspace_bounds, self.task_vars["positions"]], open(file_path, "wb"))
                self.task_vars.update({"global_timer":time.time()})

    
        if self.task_vars["task"] == "insert":
            self.current_contact_target = [0, 0, -0.5]

        return vels

    def slide(self, axis):
        """
        Task: Push down and sideways, making it slide across a plane
        """
        vels = [0, 0, 0, 0, 0, 0]

        if axis == "x":
            vels = [1, 0, 0, 0, 0, 0]
        if axis == "-x":
            vels = [-1, 0, 0, 0, 0, 0]
        if axis == "y":
            vels = [0, 1, 0, 0, 0, 0]
        if axis == "-y":
            vels = [0, -1, 0, 0, 0, 0]

        return vels

    def dist(self, a, b):
        return math.sqrt( (a - b)**2 + (a - b)**2 )

    def slide_and_test(self, axis):
        """
        Task:
        - (1) Push down and sideways, making it slide across a plane. After a time, start (2)
        - (2) Run a contact test procedure 
        """
        vels = [0, 0, 0, 0, 0, 0]
        print("Task: Slide and test")
        print("Current sub-task:", self.task_state)
        print("task time:")
        self.current_contact_target = self.standard_gravity_target
        self.maintain_override = True

        test_interval = 5
        testing_complete = True

        if self.task_state is None:
            self.task_timer = time.time()
            self.task_state = "test_contact"

        if time.time() - self.task_timer > test_interval:
            self.task_state = "test_contact"
            testing_complete = False

        if self.task_state == "test_contact":
            vels, test_done = self.test_contact(down_first=False)
            if test_done:
                testing_complete = True
                self.task_timer = time.time()
                self.task_state = "slide"
                vels = [0, 0, 0, 0, 0, 0]
        
        if self.task_state == "slide":
            vels = self.slide(axis)

        return vels


    def direct_control(self, safe=True):
        """
        Task: Wait for keyboard input to create the velocity
        - W: up
        - S: down
        - A: left
        - D: right
        - Q: left rotation
        - E: right rotation
        """
        vels = [0, 0, 0, 0, 0, 0]
        self.maintain_contact = False

        forces_safety_threshold = 10
        # (t, R)  = self.listener.lookupTransform('wam/wrist_palm_stump_link', 'wam/base_link', rospy.Time(0))
        (t, R)  = self.get_pos()
        print('ALTITUDE: ')
        print(t)
        print('-----------------')

        if "w" in self.current_pressed_keys:
            if not safe or self.delta_from_floating[2] > -forces_safety_threshold:
                vels[2] = -1
        if "s" in self.current_pressed_keys:
            if not safe or self.delta_from_floating[2] < forces_safety_threshold:
                vels[2] = 1
        if "a" in self.current_pressed_keys:
            if not safe or self.delta_from_floating[0] > -forces_safety_threshold:
                vels[0] = -1
        if "d" in self.current_pressed_keys:
            if not safe or self.delta_from_floating[0] < forces_safety_threshold:
                vels[0] = 1
        if "q" in self.current_pressed_keys:
            if not safe or (abs(self.delta_from_floating[0]) < forces_safety_threshold and abs(self.delta_from_floating[1]) < forces_safety_threshold and abs(self.delta_from_floating[2]) < forces_safety_threshold):
                vels[4] = -1
        if "e" in self.current_pressed_keys:
            if not safe or (abs(self.delta_from_floating[0]) < forces_safety_threshold and abs(self.delta_from_floating[1]) < forces_safety_threshold and abs(self.delta_from_floating[2]) < forces_safety_threshold):
                vels[4] = 1
                

        return vels

    def test_contact(self, frozen_axis="y", initial_contact_target=[None, None, None]):
        """
        Task: Perform an array of motions, and detect for each the mechanical constraints (from the force readings)
        """
        
        vels = [0, 0, 0, 0, 0, 0]
        self.current_contact_target = initial_contact_target
        
        # loop = True
        loop = False
        
        if self.task_state is None: # Setting the initial sub-task
            if initial_contact_target != [None, None, None]:
                self.task_state = "initial_contact"
            else:
                self.task_state = "test_contact"
                self.maintain_override = False
                self.maintain_contact = False

        if self.task_state  == "initial_contact":
            self.maintain_override = True

            if self.last_contact_time < 1:
                self.task_state = "test_contact"
                self.maintain_override = False
                self.maintain_contact = False

        if self.task_finished:
            print("Task finished")
            print("Found resistances:")
            pprint(self.task_vars["resistances"])
            time.sleep(999999999)
            return vels#, self.task_finished
        
        if self.task_state  == "test_contact":
            self.maintain_contact = False
            if "initPos" not in self.task_vars:
                self.task_vars.update({"initPos":self._getPos()})
                self.task_vars.update({"motion_idx":0})
                self.task_vars.update({"resistances":[[False, False], [False, False], [False, False], [False, False], [False, False], [False, False]]})
                self.task_timer = time.time()

            motions = ["x", "-x", "y", "-y", "z", "-z", "rz", "-rz"]#, "ry", "-ry"]
            # motions = ["rz"]
            # motions = ["x", "y", "z"]
            motion_duration = 5

            # Current motion done, starting new one and resetting timer
            if time.time() - self.task_timer > motion_duration:
                self._movePos(self.task_vars["initPos"])
                if self.task_vars["motion_idx"] < len(motions)-1:
                    self.task_vars.update({"motion_idx": self.task_vars["motion_idx"]+1})    
                else:
                    if loop:
                        print("Resetting motion idx to 0")
                        self.task_vars.update({"motion_idx": -1})
                        print(self.task_vars)
                    else:
                        print("All motion performed - Returning zero vels")
                        
                        self.task_finished = True
                        
                    return vels#, self.task_finished
                self.task_timer = time.time()

            # Perform a series of motion, returning to the initial position after each test.
            print("Executing sub-task:", motions[self.task_vars["motion_idx"]])
            current_motion = motions[self.task_vars["motion_idx"]]
            
            
            if current_motion == "x":
                vels[0] = 1
                ignores = [False, True, True, True, False, True]
            if current_motion == "-x":
                vels[0] = -1
                ignores = [False, True, True, True, False, True]
            if current_motion == "y":
                vels[1] = 1
                ignores = [True, False, True, False, True, True]
            if current_motion == "-y":
                vels[1] = -1
                ignores = [True, False, True, False, True, True]
            if current_motion == "z":
                vels[2] = 1
                ignores = [True, True, False, True, True, True]
            if current_motion == "-z":
                vels[2] = -1
                ignores = [True, True, False, True, True, True]
            if current_motion == "ry":
                vels[4] = 1
                # ignores = [True, True, False, True, True, True]
            if current_motion == "-ry":
                vels[4] = -1
            if current_motion == "rz":
                vels[5] = 1
                ignores = [True, True, True, True, True, False]
            if current_motion == "-rz":
                vels[5] = -1
                ignores = [True, True, True, True, True, False]
            
            new_resistance = self.detect_resistance(ignore=ignores)
            print("Real-time Resistance:")
            pprint(new_resistance)
            print("Already found resistances:")
            pprint(self.task_vars["resistances"])
            for i, resistance in enumerate(new_resistance):
                if True in resistance:
                    print("Found new resistance on axis:", i)
                    if resistance[0]:
                        self.task_vars["resistances"][i][0] = True
                    if resistance[1]:
                        self.task_vars["resistances"][i][1] = True
            # if self.task_vars["resistances"][0][0]:
            #     time.sleep(999999999999999999)
            # # Interrupting the current motion if a resistance is found to avoid wear
            # for resistance_axis in new_resistance:
            #     if "True" in resistance_axis:
            #         print("INTERRUPTING MOTION!")
            #         self.task_vars.update({"motion_idx": self.task_vars["motion_idx"]+1})
            #         self.task_timer = time.time()

        return vels#, self.task_finished

    def align_with_plane_below(self):
        """
        Task: try to regulate a side contact to 0 by rotating against it
        """
        vels = [0, 0, 0, 0, 0, 0]
        self.maintain_contact = True
        self.current_contact_target = self.standard_gravity_target
        self.maintain_override = True

        if abs(self.delta_from_floating[3]) > 0.1:
            vels[3] = self.delta_from_floating[3]*10
        # if abs(self.delta_from_floating[4]) > 0.01:
        #     vels[3] = -self.delta_from_floating[4]*10

        return vels

    def explore_laterally(self, axis):
        """
        Task: 
        - (1) Go down until contact
        - (2) Slide until side contact (go to (3)) or bottom contact is lost (go to (1))
        - (3) Climb while maintaining side contact, then switch back to (2)
        """
        vels = [0, 0, 0, 0, 0, 0]
        print("current subtask:", self.task_state)

        if self.last_contact_time < 2:
            self.task_timer = time.time()

        if self.task_state is None: # Setting the initial sub-task
            self.task_state = "down"
        
        if self.task_state  == "down":
            self.maintain_override = False
            vels = [0, 0, 1, 0, 0, 0]

            if self.last_contact_time < 1:
                self.task_state = "sliding"
                self.maintain_override = True

        if self.task_state  == "sliding":
            if axis == "x":
                vels = [1, 0, 0, 0, 0, 0]
            if axis == "-x":
                vels = [-1, 0, 0, 0, 0, 0]
            if axis == "y":
                vels = [0, 1, 0, 0, 0, 0]
            if axis == "-y":
                vels = [0, -1, 0, 0, 0, 0]
        
            # Transition to Sliding
            # If a new side contact is detected, we record the current contact force as a target, and try to maintain that while going up
            diff_x = self.force_data[0]-self.averages[0]
            diff_y = self.force_data[1]-self.averages[1]
            force_offset = 1
            if abs(diff_x) > force_offset or abs(diff_y) > force_offset:
                if abs(diff_x) > force_offset:
                    target_x = abs(diff_x)/diff_x*0.5
                else:
                    target_x = 0
                if abs(diff_y) > force_offset:
                    target_y = abs(diff_y)/diff_y*0.5
                else:
                    target_y = 0
                print("Switching sub-task to Climbing")
                self.current_contact_target = (target_x, target_y, None)
                self.task_state = "climbing"
                # self.current_task = {"name":"up", "args":{}}
                self.maintain_override = True
            
            # Transition to Down
            if self.last_contact_time > 2 and time.time() - self.task_timer > 3:
                self.task_state = "down"
                self.current_contact_target = self.standard_gravity_target
                self.maintain_override = False
            
        
        if self.task_state == "climbing":
            vels = [0, 0, -1, 0, 0, 0]
            if self.last_contact_time > 2:
                self.task_state = "sliding"
                self.last_contact_time = 0
                self.current_contact_target = self.standard_gravity_target
                self.maintain_override = False
                self.task_timer = time.time()
        print("Task timer:", time.time()-self.task_timer)
        return vels

    def orient_towards(self):
        """
        Task: Assuming the object hit a planar surface with an angle, we try to regulate the side contact to 0 by rotating and translating back (trying to rotate around the object)
        """
        print("Re-orienting!")
        # delta_from_floating_x = self.force_data[0] - self.average_x
        # delta_from_floating_y = self.force_data[1] - self.average_y

        # print("deltas from floating:", delta_from_floating_x, "/", delta_from_floating_y)

        target_lat = 0.5
        # target_y = 1.0*0.4

        # Rotation around fingertips: 20 lat for 1 rot
        # Rotation around half cylinder: 28/1 (lat/rot)
        gain_lateral = 40
        gain_rot = 1
        global_gain = 2
        if self.project_all_in_effector_space:
            gain_rot *=-1
        gain_z = 0.3
        vel_z = -0.1

        # return [gain_lateral, 0, 0, 0, gain_rot, 0]

        if abs(self.delta_from_floating[1]) > target_lat:
            print("Regulating lateral forces, target=", target_lat)
            scale =self.delta_from_floating[1]*global_gain
            vel_y = -gain_lateral*scale
            vel_rx = gain_rot*scale
            # vel_rx = delta_from_floating_y/abs(delta_from_floating_y)*gain_rot
            print("vel y:", vel_y)
            print("rot rx:", vel_rx)
        else:
            vel_rx = 0
            vel_y = 0
        
        if abs(self.delta_from_floating[0]) > target_lat:
            print("Regulating lateral forces, target=", target_lat)
            scale = self.delta_from_floating[0]*global_gain
            vel_x = -scale*gain_lateral
            vel_ry = -scale*gain_rot
            # vel_ry = -delta_from_floating_x/abs(delta_from_floating_x)*gain_rot
            print("vel x:", vel_x)
            print("rot ry:", vel_ry)
            # if delta_from_floating_y > 0:
            #     vel_y = -1*(1+abs(delta_from_floating_y))
            # else:
            #     vel_y = 1*(1+abs(delta_from_floating_y))
            
        else:
            vel_ry = 0
            vel_x = 0
        return [vel_x, vel_y, vel_z, vel_rx, vel_ry, 0]


    def apply_gains(self, vels, gains):
        for i in range(len(vels)):
            vels[i] *= gains[i]        
        return vels
    
    
    def maintain_side_contact(self, target=None, gain = 1):
        """
        Regulation task: Try to reach the lateral force targets by acting on the related velocities
        """
        print("Current contact target:", target)
        print("Last contact:", self.last_contact_time)
        # print("Average z/current_z", self.average_z, self.force_data[2])

        #if self.project_all_in_effector_space:
        #    self.delta_from_floating[2] *=-1
            # self.delta_from_floating *=-1
            # target *=-1
        # self.recorded_delta_z.append(self.delta_from_floating)
        # print("Forces:")
        # print("Current delta from floating:", self.delta_from_floating)
        # print("z force target:", target)

        precision = 0.1

        if (target[0] is None or self.dist(self.delta_from_floating[0], target[0]) < precision) and (target[1] is None or self.dist(self.delta_from_floating[1], target[1]) < precision) and (target[2] is None or self.dist(self.delta_from_floating[2], target[2]) < precision):
            self.last_contact_time = 0
        else:
            self.last_contact_time += 1/self.rate

        vels = [None, None, None]
        for i, vel_target in enumerate(target):
            if vel_target is not None:
                if vel_target is not None and self.force_data is not None:
                    vel = (vel_target-self.delta_from_floating[i])*gain
                if self.last_contact_time > 5:
                    print("No recent contact, speeding up")
                    vel *= 4
                else:
                    vel *= 0.5
                if self.project_all_in_effector_space and i == 2:
                    vel *=-1
                
                if i == 0 or i == 1:
                    vel *= 2            

                vels[i] = vel
        
        return vels
            
    def draw_cirle(self, axis=None, inverse_radius_size=3.14/1000*2):
        """
        Task: Slide across an horizontal planar surface with a cicular motion
        """
        if axis == "x":
            vel_x = 0
            vel_y = math.cos(self.t)
            vel_z = math.sin(self.t)
        if axis == "y":
            vel_x = math.cos(self.t)
            vel_y = 0
            vel_z = math.sin(self.t)
        if axis == "z":
            vel_x = math.cos(self.t)
            vel_y = math.sin(self.t)
            vel_z = 0
        
        self.t += inverse_radius_size
        scale = 100
        vel_x *= scale
        vel_y *= scale
        vel_z *= scale
        # print("vels: ", vel_y, vel_z)
        # self.publishJointVelocity_jog([0.,vel_y,vel_z, 0, 0, 0])
        return [vel_x, vel_y, vel_z, 0, 0, 0]
    
    #velocity of the 7 joints of the robot
    # def publishJointVelocity(self, vel):
    #     if len(vel) !=7:
    #         rospy.logerr('Velocity vector not of size 7')
    #         return
    #     vel = self._check_vels(vel)
    #     msg = RTJointVel()
    #     msg.velocities = vel
    #     self._robot_vel_publisher.publish(msg)
        
        
    # def publishJointVelocity_jog(self, vel):
    #     if len(vel) !=6:
    #         rospy.logerr('Velocity vector not of size 6')
    #         return

    #     msg = JogFrame()
    #     header = Header()
    #     header.frame_id = "wam/base_link"
    #     header.stamp = rospy.Time.now()
    #     msg.header = header
    #     msg.group_name = "arm"
    #     msg.link_name = "wam/wrist_palm_link"
    #     # msg.link_name = "obj_frame"
    #     msg.avoid_collisions = True
    #     vec_T = Vector3()
    #     vec_T.x = vel[0]
    #     vec_T.y = vel[1]
    #     vec_T.z = vel[2]
    #     msg.linear_delta = vec_T
    #     vec_R = Vector3()
    #     vec_R.x = vel[3]
    #     vec_R.y = vel[4]
    #     vec_R.z = vel[5]
    #     msg.angular_delta = vec_R
    #     # pprint(msg)
    #     self._robot_vel_publisher_jog.publish(msg)
        
        
    def _cart_vels(self, xyz_vel = [0.,0.,0.], rpy_vel = [0.,0.,0.]):
        move = xyz_vel+rpy_vel
        j = self._robot_joint_state
        Jac = np.asarray(self.arm_group.get_jacobian_matrix(j))
        Jac_inv = np.linalg.pinv(Jac)
        vel = np.matmul(Jac_inv ,np.asarray(move).reshape(-1,1))
        return vel
        

    def _check_vels(self, vel):
        thresh = 0.4
        vels =  [min(max(v, -thresh), thresh) for v in vel]
        mini = 0.15
        # vels[0] = min(max(vels[0], mini), -mini) 
        # return [min(max(v, mini), -mini) for v in vels]
        return vels
    
    # def _getPose(self, object = 'simple_base', origin = 'world'):
    #     (trans1, rot1)  = self.listener.lookupTransform(origin, object, rospy.Time(0))
    #     return PyKDL.Frame(PyKDL.Rotation.Quaternion(*rot1), PyKDL.Vector(*trans1))


    ##########################################
    # callbacks
    ##########################################
    
    def testCallback(self, msg):
        self.current_task = {"name":"insert_object", "args":{}}
        
    def testCallback2(self, msg):
        self.task_vars.update({"state":"insert"})
        
    def openDone(self, msg):
        self.inHandType.publish("open")
        self.task_vars.update({"state":"done"})
        self._startPos(blocking=True, idx=4) #there are three of these set up
        
    
    def joint_state_callback(self, msg):
        self._robot_joint_state = list(msg.position)
    
    def read_forces(self):
        history_size = 20 # For computing the smoothed forces

        # pprint(req)

        # self.force_data = [req.wrench.force.x, req.wrench.force.y, req.wrench.force.z, req.wrench.torque.x, req.wrench.torque.y, req.wrench.torque.z]
        self.force_data = self.controller.get_EE_wrench()
        if self.project_all_in_effector_space:
            self.force_data[2] *=1
        # self.force_data = self.rotate_force_readings(self.force_data)

        if self.record_forces:
            # pprint(self.force_data)
            # print("Recorded forces:", len(self.recorded_forces))
            self.recorded_forces.append(self.force_data)
            if len(self.recorded_forces) % self.record_every == 0:
                pickle.dump(self.recorded_forces, open(self.record_file, "wb"))
                # print("Forces saved")

        # Smoothing the readings
        if len(self.last_forces) < self.nb_samples:
            self.last_forces.append(self.force_data)
        else:
            self.last_forces.pop(0)
            self.last_forces.append(self.force_data)
        # print(self.last_forces)
        averages = [0, 0, 0, 0, 0, 0]
        for force in self.last_forces[-history_size:-1]:
            for i in range(len(averages)):
                averages[i] += force[i]
        for i in range(len(averages)):
            averages[i] /= history_size
        self.force_data = averages

        # Subtracting the calibration readings
        for i in range(len(self.force_data)):
            self.delta_from_floating[i] = self.force_data[i]-self.averages[i]
        
        if self.project_all_in_effector_space:
            self.delta_from_floating[2] *=-1
        
    def on_press(self, key):
        self.current_pressed_keys.add(key.char)
        # print('{0} pressed'.format(key))

    def on_release(self, key):
        self.current_pressed_keys.discard(key.char)
        # print('{0} release'.format(key))

    def get_vels(self):
        vels = controller.get_EE_velocity(tool_center = se3.identity())
        return ( (vels[0], vels[1], vels[2]), (vels[3], vels[4], vels[5]))
   
    def get_pos(self):
     (R, t) = controller.get_EE_transform(tool_center = se3.identity())
     return (t, R)

if __name__ == "__main__":
    # reset_start_position = True
    reset_start_position = False

    start_position_file = "start_postion_0.json"
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

    if reset_start_position:
        json.dump(current_config, open(start_position_file, "w"))

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

    controller.set_joint_config(json.load(open(start_position_file, "r")), {})
    time.sleep(2)

    PegInHoleTask(controller)

    # control the robot
    # current_config[3] += 0.2
    # controller.set_joint_config(current_config,{})
    # time.sleep(2)

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
    #current_EE_transform = controller.get_EE_transform(tool_center = se3.identity()) #transform of tool center, 
    # current_EE_transform[1][2] += 0.02 #move up 2 cm
    #controller.set_EE_transform(current_EE_transform, control_params)
    #time.sleep(10)

    # Set EE velocity, in the world frame
    #controller.set_EE_velocity([0,0,-0.01,0,0,0], control_params)
    #time.sleep(5)
    # controller.set_EE_velocity([0,0,0,0.05,0,0], control_params)
    # time.sleep(5)

    # shutdown the robot
    controller.close()


