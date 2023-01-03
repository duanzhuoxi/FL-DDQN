import math
import numpy as np
from scipy.optimize import fsolve

BASE_STATION_NUM = 6
MAX_ENERGY_QUEUE_LENGTH = 4  # in units
flat_action_num = (BASE_STATION_NUM + 1) * (MAX_ENERGY_QUEUE_LENGTH + 1)
CONTROL_ACTION_MAP = np.arange(flat_action_num). \
    reshape(BASE_STATION_NUM + 1, MAX_ENERGY_QUEUE_LENGTH + 1)

epoch_duration = 5 * 10 ** -3  # in second
channel_bandwidth = 0.6 * 10 ** 6  # in Hz
noise_power = 1.5 * 10 ** (-8)  # in Watt
# task related parameters
# input_data_size and cpu_cycles are constant in paper
# input data size of a task, in bit
input_data_size_each_task = 3 * 10 ** 4
# required CPU cycles for a task, in cycle
cpu_cycles_each_task = 8.375 * 10 ** 6
# maximum CPU-cycle frequency, in Hz
MAX_CPU_FREQUENCY = 2 * 10 ** 9
# maximum transmit power, in Watt
MAX_TRANS_POWER = 2
ONETIME_HANDOVER_DELAY = 2 * 10 ** (-3)  # in second
MEC_SERVICE_PRICE = 1
WEIGHTS_LIST = [3, 9, 5, 2, 1]
MAX_TASK_QUEUE_LENGTH = 4  # in tasks
ENERGY_UNIT = 2 * 10 ** (-3)  # in Joule
# TASK_GENERATE_PROB = 0.6  # in probability, variable in Federated Leaning setting
ENERGY_HARVEST_LAMBDA = 0.8  # in units
EFFECTIVE_SWITCHED_CAPACITANCE = 10 ** (-28)
# channel gain state, in dB
CHANNEL_GAIN_STATE_SET = \
    [-11.23, -9.37, -7.8, -6.3, -4.68, -2.08]


# calculate the handover time
def calc_handover_time(association_decision_fn, last_association_decision_fn,
                       onetime_handover_delay_fn):
    if association_decision_fn == last_association_decision_fn:
        handover_time_fn = 0
    else:
        handover_time_fn = onetime_handover_delay_fn
    return handover_time_fn


# calculate the transmitting time, if choosing edge computing
def calc_trans_time(input_data_size_each_task_fn, channel_bandwidth_fn,
                    allocated_energy_units_num_fn, energy_unit_fn, channel_state_fn,
                    noise_power_fn, max_trans_power_fn):
    # SymPy is slow, use SciPy
    allocated_energy = allocated_energy_units_num_fn * energy_unit_fn
    a_coefficient = input_data_size_each_task_fn / channel_bandwidth_fn * math.log(2)
    b_coefficient = allocated_energy * math.pow(10, channel_state_fn / 10) / noise_power_fn

    x_solve = fsolve(
        lambda x: a_coefficient * x - np.log(1 + b_coefficient * x), np.array([0, 1000]))
    trans_time_fn = 1 / np.max(x_solve)

    # if the transmit power is too high
    actual_energy_unit_consumption_fn = 0
    if allocated_energy / trans_time_fn > max_trans_power_fn:
        trans_power = max_trans_power_fn
        trans_rate = \
            channel_bandwidth_fn * \
            math.log2(1 + math.pow(10, channel_state_fn / 10) * trans_power / noise_power_fn)
        trans_time_fn = input_data_size_each_task_fn / trans_rate
        actual_energy_unit_consumption_fn = \
            math.ceil(trans_power * trans_time_fn / energy_unit_fn)

    return trans_time_fn, actual_energy_unit_consumption_fn


# return task_completion_num, queue_len_before_handling, task_fail_penalty
# determine whether tasks are done within time constrains
# note that, in this paper, only one task is handled in one epoch
def calc_task_completion_num(task_execution_delay_fn, epoch_duration_fn, task_queue_fn):
    # before tasks are processed, the length of task queue
    queue_len_before_handling_fn = len(task_queue_fn)

    # whether tasks are done within time constrains
    if epoch_duration_fn > task_execution_delay_fn > 0:
        # only one task is handled in one epoch in this paper
        task_completion_num_fn = 1
    else:
        task_completion_num_fn = 0

    if task_completion_num_fn != 0:
        # this means all tasks are done within time constrains
        # simulate queue, viz., FIFO
        for index in range(task_completion_num_fn):
            # remove all completed tasks
            task_queue_fn.pop(0)
        task_fail_penalty_fn = 0
    else:
        # if the execution of a computation task fails, the MU receives a penalty
        # we give a narrow implementation here
        # penalty coefficient is 1
        task_fail_num = task_completion_num_fn
        task_fail_penalty_fn = 1 * task_fail_num
    return task_completion_num_fn, queue_len_before_handling_fn, task_fail_penalty_fn


# calculate utility
# WEIGHTS_LIST = [3, 9, 5, 2, 1]
def calc_utility(weights_list_fn, epoch_duration_fn, task_execution_delay_fn, task_drop_num_fn,
                 queuing_delay_fn, task_fail_penalty_fn, offload_payment_fn):
    tmp = [min(epoch_duration_fn, task_execution_delay_fn),task_drop_num_fn,queuing_delay_fn,task_fail_penalty_fn,offload_payment_fn]

    utility_record_list = [weights_list_fn[0] * math.exp(-tmp[0]),
                           weights_list_fn[1] * math.exp(-tmp[1]),
                           weights_list_fn[2] * math.exp(-tmp[2])/10,
                           weights_list_fn[3] * math.exp(-tmp[3]),
                           weights_list_fn[4] * math.exp(-tmp[4])]


    utility_this_epoch = sum(utility_record_list)
    return utility_this_epoch


# actual action to flat action
def obtain_1_d_action(computation_offload_decision_fn, allocated_energy_units_num_fn, map_fn):
    flat_action = map_fn[computation_offload_decision_fn, allocated_energy_units_num_fn]
    return flat_action


# flat action to actual action
def obtain_actual_action(map_fn, control_action_fn):
    # find the index
    actual_action = np.argwhere(map_fn == control_action_fn)
    # take out the actual actions
    computation_offload_decision_fn = actual_action[0][0]
    allocated_energy_units_num_fn = actual_action[0][1]
    return computation_offload_decision_fn, allocated_energy_units_num_fn
