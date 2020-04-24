import numpy as np
import csv

import tensorflow as tf
from keras import backend as K

from controller import Controller, StateSpace, remove_files
from manager import RewardManager
import parse_action as pa

ini_state = []
final_state = []
ini_action = []
final_action = []

# create a shared session between Keras and Tensorflow
policy_sess = tf.Session()
K.set_session(policy_sess)

NUM_LAYERS = 1  # number of layers of the state space
MAX_TRIALS = 300  # maximum number of models generated

ACCURACY_BETA = 0.8  # beta value for the moving average of the accuracy
CLIP_REWARDS = 0.0  # clip rewards in the [-0.05, 0.05] range

EXPLORATION = 0.8  # high exploration for the first 1000 steps
REGULARIZATION = 1e-3  # regularization strength

CONTROLLER_CELLS = 60  # number of cells in RNN controller
EMBEDDING_DIM = 20  # dimension of the embeddings for each state
RESTORE_CONTROLLER = True  # restore controller to continue training

# clear the previous files
remove_files()

# construct a state space
state_space = StateSpace()

# add states
# state_space.add_state(name='road', values=[0, 1, 2, 3, 4])
# state_space.add_state(name='time', values=[0, 1, 2])
# state_space.add_state(name='weather', values=[0, 1, 2, 3, 4, 5])
# state_space.add_state(name='scene-tunnel', values=[0, 1])
# state_space.add_state(name='scene-intersection', values=[0, 1])
# state_space.add_state(name='scene-construction', values=[0, 1])
# state_space.add_state(name='scene-rail', values=[0, 1])
# state_space.add_state(name='scene-toll', values=[0, 1])
# state_space.add_state(name='scene-viaduct', values=[0, 1])
# state_space.add_state(name='car', values=[0, 4, 8, 12, 16, 20, 24, 28, 32, 36])
# state_space.add_state(name='motor', values=[0, 2, 4, 6])
# state_space.add_state(name='person', values=[0, 5, 10, 15, 20, 25])
# state_space.add_state(name='truck', values=[0, 4, 8, 12])
# state_space.add_state(name='tricycle', values=[0, 2, 4, 6])
# state_space.add_state(name='bus', values=[0, 2, 5, 7])
# state_space.add_state(name='truncation', values=[0, 3, 6, 9, 12, 15, 18])
# state_space.add_state(name='occlusion', values=[0, 9, 19, 28, 38, 47,  57])
# [4, 2, 5, 1, 1, 1, 1, 1, 1, 36, 6, 25, 12, 6, 7, 18, 57]

state_space.add_state(name='vehicle', values=[round(4.1*x, 4) for x in range(0, 11)])
state_space.add_state(name='person', values=[round(1.3*x, 4) for x in range(0, 11)])
state_space.add_state(name='non-motor', values=[round(1.2*x, 4) for x in range(0, 11)])
state_space.add_state(name='group', values=[round(2.0*x, 4) for x in range(0, 11)])

state_space.add_state(name='scene1', values=[round(1.0*x, 4) for x in range(-50, 51, 2)])
state_space.add_state(name='scene2', values=[round(1.0*x, 4) for x in range(-40, 51, 2)])
state_space.add_state(name='scene3', values=[round(1.0*x, 4) for x in range(-32, 54, 2)])
state_space.add_state(name='scene4', values=[round(1.0*x, 4) for x in range(-40, 41, 2)])
state_space.add_state(name='scene5', values=[round(1.0*x, 4) for x in range(-30, 41, 2)])
state_space.add_state(name='scene6', values=[round(1.0*x, 4) for x in range(-30, 35, 2)])
state_space.add_state(name='scene7', values=[round(1.0*x, 4) for x in range(-20, 30, 2)])
state_space.add_state(name='scene8', values=[round(1.0*x, 4) for x in range(-28, 35, 3)])
state_space.add_state(name='scene9', values=[round(1.0*x, 4) for x in range(-20, 26, 3)])
state_space.add_state(name='scene10', values=[round(1.0*x, 4) for x in range(-22, 24, 3)])
state_space.add_state(name='scene11', values=[round(1.0*x, 4) for x in range(-22, 43, 4)])
state_space.add_state(name='scene12', values=[round(1.0*x, 4) for x in range(-20, 31, 5)])
state_space.add_state(name='scene13', values=[round(1.0*x, 4) for x in range(-25, 33, 5)])
state_space.add_state(name='scene14', values=[round(1.0*x, 4) for x in range(-18, 31, 5)])
state_space.add_state(name='scene15', values=[round(1.0*x, 4) for x in range(-18, 20, 5)])
state_space.add_state(name='scene16', values=[round(1.0*x, 4) for x in range(-18, 29, 5)])
state_space.add_state(name='scene17', values=[round(1.0*x, 4) for x in range(-18, 23, 5)])
state_space.add_state(name='scene18', values=[round(1.0*x, 4) for x in range(-14, 23, 5)])
state_space.add_state(name='scene19', values=[round(1.0*x, 4) for x in range(-18, 23, 5)])
state_space.add_state(name='scene20', values=[round(1.0*x, 4) for x in range(-18, 23, 5)])
state_space.add_state(name='scene21', values=[round(1.0*x, 4) for x in range(-17, 29, 5)])
state_space.add_state(name='scene22', values=[round(1.0*x, 4) for x in range(-15, 21, 5)])
state_space.add_state(name='scene23', values=[round(1.0*x, 4) for x in range(-15, 21, 5)])
state_space.add_state(name='scene24', values=[round(1.0*x, 4) for x in range(-15, 31, 5)])
state_space.add_state(name='scene25', values=[round(1.0*x, 4) for x in range(-13, 23, 5)])
# [41, 0, 13, 0, 12, 0, 20, 0]

# print the state space being searched
state_space.print_state_space()

previous_res_mAP = 0.0
total_reward = 0.0

with policy_sess.as_default():
    # create the Controller and build the internal policy network
    controller = Controller(policy_sess, NUM_LAYERS, state_space,
                            reg_param=REGULARIZATION,
                            exploration=EXPLORATION,
                            controller_cells=CONTROLLER_CELLS,
                            embedding_dim=EMBEDDING_DIM,
                            restore_controller=RESTORE_CONTROLLER)

# create the Manager
manager = RewardManager(clip_rewards=CLIP_REWARDS,
                        acc_beta=ACCURACY_BETA)

# get an initial random state space if controller needs to predict an
# action from the initial state
state = state_space.get_random_state_space(NUM_LAYERS)
print("Initial Random State : ", state_space.parse_state_space_list(state))
print()


# train for number of trails
for trial in range(MAX_TRIALS):
    with policy_sess.as_default():
        K.set_session(policy_sess)
        actions = controller.get_action(state)  # get an action for the previous state
        # if trial == 3:
        #     ini_action = actions
        # if trial == 3005:
        #     final_action = actions

    # print the action probabilities
    state_space.print_actions(actions)
    print("Predicted actions : ", state_space.parse_state_space_list(actions))

    # search data_index and get reward and res_map from the manager
    # _, rank_index = pa.k_near2(action=state_space.parse_state_space_list(actions))
    rank_index = pa.re_rank(action=state_space.parse_state_space_list(actions))
    # rank_index = [x for x in range(7000)]
    reward, previous_res_mAP = manager.get_reward(rank_index)
    print("Rewards : ", reward, "Res_mAP : ", previous_res_mAP)

    with policy_sess.as_default():
        K.set_session(policy_sess)

        total_reward += reward
        print("Total reward : ", total_reward)

        # actions and states are equivalent, save the state and reward
        state = actions
        controller.store_rollout(state, reward)

        # train the controller on the saved state and the discounted rewards
        loss = controller.train_step()
        print("Trial %d: Controller loss : %0.6f" % (trial + 1, loss))

        # if trial == 3:
        #     ini_state = state_space.parse_state_space_list(state)
        # if trial == 3005:
        #     final_state = state_space.parse_state_space_list(state)

        # write the results of this trial into a file
        # actions_test = controller.get_action(state, explore_flag=False)
        # _, rank_index_test = pa.k_near2(action=state_space.parse_state_space_list(actions_test))
        # _, previous_res_mAP_test = manager.get_reward(rank_index_test)

        with open('train_history.csv', mode='a+') as f:
            data = [total_reward, previous_res_mAP, reward]  # , previous_res_mAP_test
            data.extend(state_space.parse_state_space_list(state))
            writer = csv.writer(f)
            writer.writerow(data)
    print()

with policy_sess.as_default():
    K.set_session(policy_sess)
    actions = controller.get_action(state, explore_flag=False)
    with open('final_state.csv', 'a', newline='') as f:
        data = []
        data.extend(state_space.parse_state_space_list(actions))
        writer = csv.writer(f)
        writer.writerow(data)
    print('final_state:', state_space.parse_state_space_list(actions))

# print("Total Reward : ", total_reward)
# print("ini_state:", ini_state)
# state_space.print_actions(ini_action)
# print("final_state:", final_state)
# state_space.print_actions(final_action)
