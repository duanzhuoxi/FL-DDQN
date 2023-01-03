from Enviroment import *
from Task import Task
import random
from scipy.stats import poisson as scipy_poisson
from scipy.stats import bernoulli as scipy_bernoulli


class Client:
    def __init__(self, global_rl_agent, global_model_weight, client_id):
        self.rl_agent = global_rl_agent
        self.rl_agent.model.set_weights(global_model_weight)
        self.utility_hist = []
        # task_execution_delay,task_drop_num, queuing_delay, task_fail_penalty, offload_payment
        self.task_execution_delay_hist = []
        self.task_drop_num_hist = []
        self.queuing_delay_hist = []
        self.task_fail_penalty_hist = []
        self.offload_payment_hist = []
        self.energy_hist = []
        self.loss_hist = []
        self.ID = client_id

    # 从服务器 更新Client的模型权重
    def update_model_from_server(self, global_model_weight):
        self.rl_agent.model.set_weights(global_model_weight)
        # 注意不要忘了更新target model
        self.rl_agent.update_target_model()

    def train(self, epoch_num, task_generate_prob):
        # before the process begins,
        # there should be initial energy and task
        # not mentioned in paper
        task_queue = []
        energy_queue = 1
        task = Task()
        task.data_size = input_data_size_each_task
        task.cpu_cycles = cpu_cycles_each_task
        task_queue.append(task)

        # begin the process simulation
        # all parameters are based on experiment1
        # assume the initial offload decision
        computation_offload_decision = 0
        allocated_energy_units_num = 1
        # obtain the 1-D action
        control_action = obtain_1_d_action(
            computation_offload_decision, allocated_energy_units_num, CONTROL_ACTION_MAP)
        association_decision = 1
        last_association_decision = association_decision
        # the network(channel) state are fluctuated,
        # the transitions of which happen across the discrete decision epochs
        # following respective randomly generated matrices
        base_station_channel_state = []  # network states among all base stations at current epoch
        for base_station_index in range(BASE_STATION_NUM):
            base_station_channel_state.append(random.sample(CHANNEL_GAIN_STATE_SET, 1)[0])

        # record the observation
        observation_raw = [len(task_queue), energy_queue, association_decision]
        observation_raw.extend(base_station_channel_state)
        observation = np.array(observation_raw)

        cur_loss = 0
        for epoch_index in range(epoch_num):
            # obtain the task execution delay
            task_execution_delay = 0
            local_time_consumption = 0
            edge_time_consumption = 0
            edge_trans_time = 0
            handover_time = 0
            # assume the execution time in the edge server is a sufficiently small constant
            edge_execution_time = 10 ** (-6)  # in second
            task_completion_num = 0
            task_fail_penalty = 0
            queuing_delay = len(task_queue)
            offload_payment = 0

            # whether the task is handled or not, the energy and the task will arrive
            # at the end of epochs, MU acquire the energy units
            # assume in this paper, the received energy units across the time horizon take integer values
            attempt_acquire_energy_units_num = scipy_poisson.rvs(ENERGY_HARVEST_LAMBDA, size=1)[0]
            # generate the task, 0 or 1
            task_arrival_indicator = scipy_bernoulli.rvs(task_generate_prob, size=1)[0]

            # if the task queue is not empty
            # and there are enough energy units for allocation, the task could be processed
            if len(task_queue) != 0 and \
                    allocated_energy_units_num <= energy_queue:
                if allocated_energy_units_num > 0 and computation_offload_decision == 0:
                    # local computing
                    # When a computation task is scheduled for processing locally
                    # at the mobile device of the MU,
                    # calculate the allocated frequency
                    attempt_allocated_cpu_frequency = math.sqrt(
                        allocated_energy_units_num * ENERGY_UNIT /
                        (EFFECTIVE_SWITCHED_CAPACITANCE * cpu_cycles_each_task)
                    )
                    # actual consumed energy units might not equal to the allocated one
                    # due to the constrains MAX_CPU_FREQUENCY
                    allocated_cpu_frequency = 0
                    if attempt_allocated_cpu_frequency >= MAX_CPU_FREQUENCY:
                        allocated_cpu_frequency = MAX_CPU_FREQUENCY
                        allocated_energy_units_num = \
                            math.ceil(
                                math.pow(MAX_CPU_FREQUENCY, 2) *
                                EFFECTIVE_SWITCHED_CAPACITANCE * cpu_cycles_each_task / ENERGY_UNIT)
                    else:
                        allocated_cpu_frequency = attempt_allocated_cpu_frequency

                    # the time needed for local computation task execution
                    local_computation_task_execution_time = \
                        cpu_cycles_each_task / allocated_cpu_frequency
                    # for clarity
                    local_time_consumption = local_computation_task_execution_time
                    task_execution_delay = local_time_consumption
                elif allocated_energy_units_num > 0 and computation_offload_decision != 0:
                    # edge computing
                    # considering handover time
                    handover_time = calc_handover_time(
                        association_decision, last_association_decision, ONETIME_HANDOVER_DELAY)

                    # the time of transmitting task input data to edge
                    edge_trans_time, actual_energy_unit_consumption = calc_trans_time(
                        input_data_size_each_task, channel_bandwidth,
                        allocated_energy_units_num, ENERGY_UNIT,
                        base_station_channel_state[association_decision - 1],  # noted association_decision - 1
                        noise_power, MAX_TRANS_POWER)

                    # if the MAX_TRANS_POWER is being reached,
                    # obtain the actual energy unit consumption
                    if actual_energy_unit_consumption != 0:
                        allocated_energy_units_num = actual_energy_unit_consumption

                    # the time needed for a task execution in edge.
                    # include handover time, the time of transmitting task input data to edge,
                    # and edge_execution_time
                    edge_time_consumption = sum([handover_time, edge_trans_time, edge_execution_time])
                    task_execution_delay = edge_time_consumption

                    # a payment is required for the access to MEC service
                    # when the MU decides to offload a computation task for MEC server execution
                    # local computing, the payment is 0
                    mec_service_duration = min(task_execution_delay, epoch_duration) - handover_time
                    offload_payment = MEC_SERVICE_PRICE * mec_service_duration
                elif allocated_energy_units_num == 0:
                    # no energy units are allocated
                    task_execution_delay = 0
                else:
                    # in fact, the "else" is redundant
                    task_execution_delay = 0

                # to determine if the task is done
                # in this paper, only one task being handled in one epoch
                # i.e., task_completion_num = 1 or 0
                task_completion_num, queue_len_before_handling, task_fail_penalty = \
                    calc_task_completion_num(task_execution_delay, epoch_duration, task_queue)

                energy_queue = energy_queue - allocated_energy_units_num

            # update the energy queue for the next epoch
            # if the task is processed in real
            energy_queue = \
                min([energy_queue + attempt_acquire_energy_units_num,
                     MAX_ENERGY_QUEUE_LENGTH])

            # treat the queuing delay during a decision epoch j equivalently as the length of a task queue
            # if tasks are completed in this epoch, the delay should be removed
            queuing_delay = len(task_queue)

            # calculate the number of task drop
            task_drop_num = \
                max(len(task_queue) - task_completion_num + task_arrival_indicator - MAX_TASK_QUEUE_LENGTH, 0)

            # update task queue
            if task_arrival_indicator != 0 and len(task_queue) < MAX_TASK_QUEUE_LENGTH:
                # there are task arrival and queue space
                # considering future more generated tasks
                for task_arrival_index in range(task_arrival_indicator):
                    task = Task()
                    task.data_size = input_data_size_each_task
                    task.cpu_cycles = cpu_cycles_each_task
                    task_queue.append(task)
                    # MAX_TASK_QUEUE_LENGTH constrains
                    if len(task_queue) >= MAX_TASK_QUEUE_LENGTH:
                        break

            # calculate reward (utility), especially if the task (and energy) queue is not empty
            # in my opinion, when the task queue is empty, energy should not be allocated
            # however, in this paper, it should be learned
            utility = calc_utility(WEIGHTS_LIST, epoch_duration, task_execution_delay,
                                   task_drop_num, queuing_delay, task_fail_penalty, offload_payment)
            self.task_execution_delay_hist.append(task_execution_delay)
            self.task_drop_num_hist.append(task_drop_num)
            self.queuing_delay_hist.append(queuing_delay)
            self.task_fail_penalty_hist.append(task_fail_penalty)
            self.offload_payment_hist.append(offload_payment)
            self.energy_hist.append(allocated_energy_units_num)

            # the network(channel) state are fluctuated,
            # the transitions of which happen across the discrete decision epochs
            # following respective randomly generated matrices
            base_station_channel_state = []  # network states among all base stations at current epoch
            for base_station_index in range(BASE_STATION_NUM):
                base_station_channel_state.append(random.sample(CHANNEL_GAIN_STATE_SET, 1)[0])

            # the MU makes a joint control action for next epoch
            # noted that, in this paper, actions are two-dimension (not the same as traditional DQN)
            # hence, we need to find a map
            next_control_action = self.rl_agent.act(observation)
            next_computation_offload_decision, next_allocated_energy_units_num = \
                obtain_actual_action(CONTROL_ACTION_MAP, next_control_action)

            # record association decision in the last epoch
            last_association_decision = association_decision

            # obtain and update the base station association
            if computation_offload_decision != 0:
                # offload to a base station
                association_decision = computation_offload_decision
            elif computation_offload_decision == 0 and last_association_decision != 0:
                # current task is processed in local,
                # but the last one was processed in a certain base station
                association_decision = last_association_decision
            else:
                # for coherence
                association_decision = last_association_decision

            # store states into replay memory
            # record the next observation
            observation_raw_ = [len(task_queue), energy_queue, association_decision]
            observation_raw_.extend(base_station_channel_state)
            observation_ = np.array(observation_raw_)
            # according to computation_offload_decision and allocated_energy_units_num
            # get an one-dimension action
            self.rl_agent.remember(observation, control_action, utility, observation_)
            # update the last observation
            observation = observation_

            # if there are enough transitions, perform learning
            # at least the size of one mini_batch
            if self.rl_agent.memory_counter >= self.rl_agent.mini_batch_size:
                cur_loss = self.rl_agent.replay()
                self.loss_hist.append(cur_loss)

            # record the reward for later analysis
            self.utility_hist.append(utility)

            # update the control action for next epoch
            control_action = next_control_action
            computation_offload_decision = next_computation_offload_decision
            allocated_energy_units_num = next_allocated_energy_units_num

        return \
            self.rl_agent.model.get_weights(), \
            self.rl_agent.learning_step, \
            self.rl_agent.memory_counter, \
            self.rl_agent.memory

    def train2(self, epoch_num, task_generate_prob):
        # computing offload model
        BASE_STATION_NUM = 6
        epoch_duration = 5 * 10 ** -3  # in second
        channel_bandwidth = 0.6 * 10 ** 6  # in Hz
        noise_power = 1.5 * 10 ** (-8)  # in Watt
        # maybe could change this parameter
        # or the transmit power is always higher than 2 Watt
        EPOCH_NUM = epoch_num

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
        MAX_ENERGY_QUEUE_LENGTH = 4  # in units
        ENERGY_UNIT = 2 * 10 ** (-3)  # in Joule
        task_queue = []
        # energy_queue = 0
        task_drop_count = 0
        TASK_GENERATE_PROB = task_generate_prob  # in probability
        ENERGY_HARVEST_LAMBDA = 0.8  # in units
        EFFECTIVE_SWITCHED_CAPACITANCE = 10 ** (-28)
        # channel gain state, in dB
        CHANNEL_GAIN_STATE_SET = \
            [-11.23, -9.37, -7.8, -6.3, -4.68, -2.08]
        edge_execution_time = 10 ** (-6)  # in second

        # before the process begins,
        # there should be initial energy and task
        # not mentioned in paper
        energy_queue = 1
        task = Task()
        task.data_size = input_data_size_each_task
        task.cpu_cycles = cpu_cycles_each_task
        task_queue.append(task)

        # record the utility when the network is converged
        greedy_execution_utility = []

        # update the control action for next epoch
        # server execution
        task_execution_delay_list = []
        allocated_energy_units_num_list = []
        allocated_energy_units_num = energy_queue
        computation_offload_decision = 1
        # begin the process simulation
        # assume the initial offload decision
        association_decision = computation_offload_decision
        last_association_decision = association_decision
        # the network(channel) state are fluctuated,
        # the transitions of which happen across the discrete decision epochs
        # following respective randomly generated matrices
        base_station_channel_state = []  # network states among all base stations at current epoch
        for base_station_index in range(BASE_STATION_NUM):
            base_station_channel_state.append(random.sample(CHANNEL_GAIN_STATE_SET, 1)[0])

        if energy_queue != 0 and len(task_queue) != 0:
            # local computing
            # When a computation task is scheduled for processing locally
            # at the mobile device of the MU,
            # calculate the allocated frequency
            attempt_allocated_cpu_frequency = math.sqrt(
                allocated_energy_units_num * ENERGY_UNIT /
                (EFFECTIVE_SWITCHED_CAPACITANCE * cpu_cycles_each_task)
            )
            # actual consumed energy units might not equal to the allocated one
            # due to the constrains MAX_CPU_FREQUENCY
            allocated_cpu_frequency = 0
            if attempt_allocated_cpu_frequency >= MAX_CPU_FREQUENCY:
                allocated_cpu_frequency = MAX_CPU_FREQUENCY
                allocated_energy_units_num = \
                    math.ceil(
                        math.pow(MAX_CPU_FREQUENCY, 2) *
                        EFFECTIVE_SWITCHED_CAPACITANCE * cpu_cycles_each_task / ENERGY_UNIT
                    )
            else:
                allocated_cpu_frequency = attempt_allocated_cpu_frequency

            # the time needed for local computation task execution
            local_computation_task_execution_time = \
                cpu_cycles_each_task / allocated_cpu_frequency
            # for clarity
            local_time_consumption = local_computation_task_execution_time
            task_execution_delay = local_time_consumption
            task_execution_delay_list.append(task_execution_delay)
            allocated_energy_units_num_list.append(allocated_energy_units_num)

            # edge computing
            for bs_index in range(BASE_STATION_NUM):
                # edge computing
                # considering handover time
                handover_time = calc_handover_time(
                    association_decision, last_association_decision, ONETIME_HANDOVER_DELAY)

                # the time of transmitting task input data to edge for each base station
                edge_trans_time, actual_energy_unit_consumption = calc_trans_time(
                    input_data_size_each_task, channel_bandwidth,
                    energy_queue, ENERGY_UNIT,  # with maximum possible energy units in the energy queue
                    base_station_channel_state[bs_index],
                    noise_power, MAX_TRANS_POWER)

                # if the MAX_TRANS_POWER is being reached,
                # obtain the actual energy unit consumption
                if actual_energy_unit_consumption != 0:
                    allocated_energy_units_num = actual_energy_unit_consumption

                # the time needed for a task execution in edge.
                # include handover time, the time of transmitting task input data to edge,
                # and edge_execution_time
                edge_time_consumption = sum([handover_time, edge_trans_time, edge_execution_time])
                task_execution_delay = edge_time_consumption
                task_execution_delay_list.append(task_execution_delay)
                allocated_energy_units_num_list.append(allocated_energy_units_num)

            # selects a BS that achieves the minimum task execution delay
            # to offload the input data of a scheduled computation task for MEC server execution
            computation_offload_decision = task_execution_delay_list.index(min(task_execution_delay_list))
            # obtain the corresponding allocated_energy_units_num
            allocated_energy_units_num = allocated_energy_units_num_list[computation_offload_decision]

        for epoch_index in range(EPOCH_NUM):
            # obtain the task execution delay
            task_execution_delay = 0
            local_time_consumption = 0
            edge_time_consumption = 0
            edge_trans_time = 0
            handover_time = 0
            # assume the execution time in the edge server is a sufficiently small constant
            task_completion_num = 0
            task_fail_penalty = 0
            queuing_delay = len(task_queue)
            offload_payment = 0

            # whether the task is handled or not, the energy and the task will arrive
            # at the end of epochs, MU acquire the energy units
            # assume in this paper, the received energy units across the time horizon take integer values
            attempt_acquire_energy_units_num = scipy_poisson.rvs(ENERGY_HARVEST_LAMBDA, size=1)[0]
            # generate the task, 0 or 1
            task_arrival_indicator = scipy_bernoulli.rvs(TASK_GENERATE_PROB, size=1)[0]

            # if the task queue is not empty
            # and there are enough energy units for allocation, the task could be processed
            if len(task_queue) != 0 and \
                    allocated_energy_units_num <= energy_queue:
                if allocated_energy_units_num > 0 and computation_offload_decision == 0:
                    # local computing
                    # When a computation task is scheduled for processing locally
                    # at the mobile device of the MU,
                    # calculate the allocated frequency
                    attempt_allocated_cpu_frequency = math.sqrt(
                        allocated_energy_units_num * ENERGY_UNIT /
                        (EFFECTIVE_SWITCHED_CAPACITANCE * cpu_cycles_each_task)
                    )
                    # actual consumed energy units might not equal to the allocated one
                    # due to the constrains MAX_CPU_FREQUENCY
                    allocated_cpu_frequency = 0
                    if attempt_allocated_cpu_frequency >= MAX_CPU_FREQUENCY:
                        allocated_cpu_frequency = MAX_CPU_FREQUENCY
                        allocated_energy_units_num = \
                            math.ceil(
                                math.pow(MAX_CPU_FREQUENCY, 2) *
                                EFFECTIVE_SWITCHED_CAPACITANCE * cpu_cycles_each_task / ENERGY_UNIT)
                    else:
                        allocated_cpu_frequency = attempt_allocated_cpu_frequency

                    # the time needed for local computation task execution
                    local_computation_task_execution_time = \
                        cpu_cycles_each_task / allocated_cpu_frequency
                    # for clarity
                    local_time_consumption = local_computation_task_execution_time
                    task_execution_delay = local_time_consumption
                elif allocated_energy_units_num > 0 and computation_offload_decision != 0:
                    # edge computing
                    # considering handover time
                    handover_time = calc_handover_time(
                        association_decision, last_association_decision, ONETIME_HANDOVER_DELAY)

                    # the time of transmitting task input data to edge
                    edge_trans_time, actual_energy_unit_consumption = calc_trans_time(
                        input_data_size_each_task, channel_bandwidth,
                        allocated_energy_units_num, ENERGY_UNIT,
                        base_station_channel_state[association_decision - 1],  # noted association_decision - 1
                        noise_power, MAX_TRANS_POWER)

                    # if the MAX_TRANS_POWER is being reached,
                    # obtain the actual energy unit consumption
                    if actual_energy_unit_consumption != 0:
                        allocated_energy_units_num = actual_energy_unit_consumption

                    # the time needed for a task execution in edge.
                    # include handover time, the time of transmitting task input data to edge,
                    # and edge_execution_time
                    edge_time_consumption = sum([handover_time, edge_trans_time, edge_execution_time])
                    task_execution_delay = edge_time_consumption

                    # a payment is required for the access to MEC service
                    # when the MU decides to offload a computation task for MEC server execution
                    # local computing, the payment is 0
                    mec_service_duration = min(task_execution_delay, epoch_duration) - handover_time
                    offload_payment = MEC_SERVICE_PRICE * mec_service_duration
                elif allocated_energy_units_num == 0:
                    # no energy units are allocated
                    task_execution_delay = 0
                else:
                    # in fact, the "else" is redundant
                    task_execution_delay = 0

                # to determine if the task is done
                # in this paper, only one task being handled in one epoch
                # i.e., task_completion_num = 1 or 0
                task_completion_num, queue_len_before_handling, task_fail_penalty = \
                    calc_task_completion_num(task_execution_delay, epoch_duration, task_queue)

                energy_queue = energy_queue - allocated_energy_units_num

            # update the energy queue for the next epoch
            # if the task is processed in real
            energy_queue = \
                min([energy_queue + attempt_acquire_energy_units_num,
                     MAX_ENERGY_QUEUE_LENGTH])

            # treat the queuing delay during a decision epoch j equivalently as the length of a task queue
            # if tasks are completed in this epoch, the delay should be removed
            queuing_delay = len(task_queue)

            # calculate the number of task drop
            task_drop_num = \
                max(len(task_queue) - task_completion_num + task_arrival_indicator - MAX_TASK_QUEUE_LENGTH, 0)

            # record the total number of task drop
            task_drop_count = task_drop_count + task_drop_num

            # update task queue
            if task_arrival_indicator != 0 and len(task_queue) < MAX_TASK_QUEUE_LENGTH:
                # there are task arrival and queue space
                # considering future more generated tasks
                for task_arrival_index in range(task_arrival_indicator):
                    task = Task()
                    task.data_size = input_data_size_each_task
                    task.cpu_cycles = cpu_cycles_each_task
                    task_queue.append(task)
                    # MAX_TASK_QUEUE_LENGTH constrains
                    if len(task_queue) >= MAX_TASK_QUEUE_LENGTH:
                        break

            # calculate reward (utility), especially if the task (and energy) queue is not empty
            # in my opinion, when the task queue is empty, energy should not be allocated
            # however, in this paper, it should be learned
            utility = calc_utility(WEIGHTS_LIST, epoch_duration, task_execution_delay,
                                   task_drop_num, queuing_delay, task_fail_penalty, offload_payment)
            self.utility_hist.append(utility)

            self.task_execution_delay_hist.append(task_execution_delay)
            self.task_drop_num_hist.append(task_drop_num)
            self.queuing_delay_hist.append(queuing_delay)
            self.task_fail_penalty_hist.append(task_fail_penalty)
            self.offload_payment_hist.append(offload_payment)
            self.energy_hist.append(allocated_energy_units_num)

            # after the convergence of network, record the utility for analysis
            greedy_execution_utility.append(utility)

            # the network(channel) state are fluctuated,
            # the transitions of which happen across the discrete decision epochs
            # following respective randomly generated matrices
            base_station_channel_state = []  # network states among all base stations at current epoch
            for base_station_index in range(BASE_STATION_NUM):
                base_station_channel_state.append(random.sample(CHANNEL_GAIN_STATE_SET, 1)[0])

            # update the control action for next epoch
            # greedy execution
            if energy_queue != 0 and len(task_queue) != 0:
                task_execution_delay_list = []
                allocated_energy_units_num_list = []

                # local computing
                # When a computation task is scheduled for processing locally
                # at the mobile device of the MU,
                # calculate the allocated frequency
                attempt_allocated_cpu_frequency = math.sqrt(
                    energy_queue * ENERGY_UNIT /
                    (EFFECTIVE_SWITCHED_CAPACITANCE * cpu_cycles_each_task)
                )
                # actual consumed energy units might not equal to the allocated one
                # due to the constrains MAX_CPU_FREQUENCY
                allocated_cpu_frequency = 0
                if attempt_allocated_cpu_frequency >= MAX_CPU_FREQUENCY:
                    allocated_cpu_frequency = MAX_CPU_FREQUENCY
                    allocated_energy_units_num = \
                        math.ceil(
                            math.pow(MAX_CPU_FREQUENCY, 2) *
                            EFFECTIVE_SWITCHED_CAPACITANCE * cpu_cycles_each_task / ENERGY_UNIT)
                else:
                    allocated_cpu_frequency = attempt_allocated_cpu_frequency
                # the time needed for local computation task execution
                local_computation_task_execution_time = \
                    cpu_cycles_each_task / allocated_cpu_frequency
                # for clarity
                local_time_consumption = local_computation_task_execution_time
                task_execution_delay = local_time_consumption
                task_execution_delay_list.append(task_execution_delay)
                allocated_energy_units_num_list.append(allocated_energy_units_num)

                for bs_index in range(BASE_STATION_NUM):
                    # edge computing
                    # considering handover time
                    handover_time = calc_handover_time(
                        bs_index + 1, last_association_decision, ONETIME_HANDOVER_DELAY)

                    # the time of transmitting task input data to edge for each base station
                    edge_trans_time, actual_energy_unit_consumption = calc_trans_time(
                        input_data_size_each_task, channel_bandwidth,
                        energy_queue, ENERGY_UNIT,  # with maximum possible energy units in the energy queue
                        base_station_channel_state[bs_index],
                        noise_power, MAX_TRANS_POWER)

                    # if the MAX_TRANS_POWER is being reached,
                    # obtain the actual energy unit consumption
                    if actual_energy_unit_consumption != 0:
                        allocated_energy_units_num = actual_energy_unit_consumption

                    # the time needed for a task execution in edge.
                    # include handover time, the time of transmitting task input data to edge,
                    # and edge_execution_time
                    edge_time_consumption = sum([handover_time, edge_trans_time, edge_execution_time])
                    task_execution_delay = edge_time_consumption
                    task_execution_delay_list.append(task_execution_delay)
                    allocated_energy_units_num_list.append(allocated_energy_units_num)

                # execute a computation task at its own mobile device or offload it
                # to the MEC server for processing with the aim of minimizing the immediate task execution delay
                next_computation_offload_decision = task_execution_delay_list.index(min(task_execution_delay_list))
                # obtain the corresponding allocated_energy_units_num
                next_allocated_energy_units_num = allocated_energy_units_num_list[computation_offload_decision]
            else:
                next_computation_offload_decision = association_decision
                next_allocated_energy_units_num = 0

            # record association decision in the last epoch
            last_association_decision = association_decision

            # obtain and update the base station association
            if computation_offload_decision != 0:
                # offload to a base station
                association_decision = computation_offload_decision
            elif computation_offload_decision == 0 and last_association_decision != 0:
                # current task is processed in local,
                # but the last one was processed in a certain base station
                association_decision = last_association_decision
            else:
                # for coherence
                association_decision = last_association_decision

            computation_offload_decision = next_computation_offload_decision
            allocated_energy_units_num = next_allocated_energy_units_num

        average_utility = \
            sum(greedy_execution_utility) / len(greedy_execution_utility)

        return average_utility

    def train_energy(self, epoch_num, enengy_prob):
        # before the process begins,
        # there should be initial energy and task
        # not mentioned in paper
        task_generate_prob = 0.5
        energy_generate = enengy_prob
        task_queue = []
        energy_queue = 1
        task = Task()
        task.data_size = input_data_size_each_task
        task.cpu_cycles = cpu_cycles_each_task
        task_queue.append(task)

        # begin the process simulation
        # all parameters are based on experiment1
        # assume the initial offload decision
        computation_offload_decision = 0
        allocated_energy_units_num = 1
        # obtain the 1-D action
        control_action = obtain_1_d_action(
            computation_offload_decision, allocated_energy_units_num, CONTROL_ACTION_MAP)
        association_decision = 1
        last_association_decision = association_decision
        # the network(channel) state are fluctuated,
        # the transitions of which happen across the discrete decision epochs
        # following respective randomly generated matrices
        base_station_channel_state = []  # network states among all base stations at current epoch
        for base_station_index in range(BASE_STATION_NUM):
            base_station_channel_state.append(random.sample(CHANNEL_GAIN_STATE_SET, 1)[0])

        # record the observation
        observation_raw = [len(task_queue), energy_queue, association_decision]
        observation_raw.extend(base_station_channel_state)
        observation = np.array(observation_raw)

        cur_loss = 0
        for epoch_index in range(epoch_num):
            # obtain the task execution delay
            task_execution_delay = 0
            local_time_consumption = 0
            edge_time_consumption = 0
            edge_trans_time = 0
            handover_time = 0
            # assume the execution time in the edge server is a sufficiently small constant
            edge_execution_time = 10 ** (-6)  # in second
            task_completion_num = 0
            task_fail_penalty = 0
            queuing_delay = len(task_queue)
            offload_payment = 0

            # whether the task is handled or not, the energy and the task will arrive
            # at the end of epochs, MU acquire the energy units
            # assume in this paper, the received energy units across the time horizon take integer values
            attempt_acquire_energy_units_num = scipy_poisson.rvs(energy_generate, size=1)[0]
            # generate the task, 0 or 1
            task_arrival_indicator = scipy_bernoulli.rvs(task_generate_prob, size=1)[0]

            # if the task queue is not empty
            # and there are enough energy units for allocation, the task could be processed
            if len(task_queue) != 0 and \
                    allocated_energy_units_num <= energy_queue:
                if allocated_energy_units_num > 0 and computation_offload_decision == 0:
                    # local computing
                    # When a computation task is scheduled for processing locally
                    # at the mobile device of the MU,
                    # calculate the allocated frequency
                    attempt_allocated_cpu_frequency = math.sqrt(
                        allocated_energy_units_num * ENERGY_UNIT /
                        (EFFECTIVE_SWITCHED_CAPACITANCE * cpu_cycles_each_task)
                    )
                    # actual consumed energy units might not equal to the allocated one
                    # due to the constrains MAX_CPU_FREQUENCY
                    allocated_cpu_frequency = 0
                    if attempt_allocated_cpu_frequency >= MAX_CPU_FREQUENCY:
                        allocated_cpu_frequency = MAX_CPU_FREQUENCY
                        allocated_energy_units_num = \
                            math.ceil(
                                math.pow(MAX_CPU_FREQUENCY, 2) *
                                EFFECTIVE_SWITCHED_CAPACITANCE * cpu_cycles_each_task / ENERGY_UNIT)
                    else:
                        allocated_cpu_frequency = attempt_allocated_cpu_frequency

                    # the time needed for local computation task execution
                    local_computation_task_execution_time = \
                        cpu_cycles_each_task / allocated_cpu_frequency
                    # for clarity
                    local_time_consumption = local_computation_task_execution_time
                    task_execution_delay = local_time_consumption
                elif allocated_energy_units_num > 0 and computation_offload_decision != 0:
                    # edge computing
                    # considering handover time
                    handover_time = calc_handover_time(
                        association_decision, last_association_decision, ONETIME_HANDOVER_DELAY)

                    # the time of transmitting task input data to edge
                    edge_trans_time, actual_energy_unit_consumption = calc_trans_time(
                        input_data_size_each_task, channel_bandwidth,
                        allocated_energy_units_num, ENERGY_UNIT,
                        base_station_channel_state[association_decision - 1],  # noted association_decision - 1
                        noise_power, MAX_TRANS_POWER)

                    # if the MAX_TRANS_POWER is being reached,
                    # obtain the actual energy unit consumption
                    if actual_energy_unit_consumption != 0:
                        allocated_energy_units_num = actual_energy_unit_consumption

                    # the time needed for a task execution in edge.
                    # include handover time, the time of transmitting task input data to edge,
                    # and edge_execution_time
                    edge_time_consumption = sum([handover_time, edge_trans_time, edge_execution_time])
                    task_execution_delay = edge_time_consumption

                    # a payment is required for the access to MEC service
                    # when the MU decides to offload a computation task for MEC server execution
                    # local computing, the payment is 0
                    mec_service_duration = min(task_execution_delay, epoch_duration) - handover_time
                    offload_payment = MEC_SERVICE_PRICE * mec_service_duration
                elif allocated_energy_units_num == 0:
                    # no energy units are allocated
                    task_execution_delay = 0
                else:
                    # in fact, the "else" is redundant
                    task_execution_delay = 0

                # to determine if the task is done
                # in this paper, only one task being handled in one epoch
                # i.e., task_completion_num = 1 or 0
                task_completion_num, queue_len_before_handling, task_fail_penalty = \
                    calc_task_completion_num(task_execution_delay, epoch_duration, task_queue)

                energy_queue = energy_queue - allocated_energy_units_num

            # update the energy queue for the next epoch
            # if the task is processed in real
            energy_queue = \
                min([energy_queue + attempt_acquire_energy_units_num,
                     MAX_ENERGY_QUEUE_LENGTH])

            # treat the queuing delay during a decision epoch j equivalently as the length of a task queue
            # if tasks are completed in this epoch, the delay should be removed
            queuing_delay = len(task_queue)

            # calculate the number of task drop
            task_drop_num = \
                max(len(task_queue) - task_completion_num + task_arrival_indicator - MAX_TASK_QUEUE_LENGTH, 0)

            # update task queue
            if task_arrival_indicator != 0 and len(task_queue) < MAX_TASK_QUEUE_LENGTH:
                # there are task arrival and queue space
                # considering future more generated tasks
                for task_arrival_index in range(task_arrival_indicator):
                    task = Task()
                    task.data_size = input_data_size_each_task
                    task.cpu_cycles = cpu_cycles_each_task
                    task_queue.append(task)
                    # MAX_TASK_QUEUE_LENGTH constrains
                    if len(task_queue) >= MAX_TASK_QUEUE_LENGTH:
                        break

            # calculate reward (utility), especially if the task (and energy) queue is not empty
            # in my opinion, when the task queue is empty, energy should not be allocated
            # however, in this paper, it should be learned
            utility = calc_utility(WEIGHTS_LIST, epoch_duration, task_execution_delay,
                                   task_drop_num, queuing_delay, task_fail_penalty, offload_payment)
            self.task_execution_delay_hist.append(task_execution_delay)
            self.task_drop_num_hist.append(task_drop_num)
            self.queuing_delay_hist.append(queuing_delay)
            self.task_fail_penalty_hist.append(task_fail_penalty)
            self.offload_payment_hist.append(offload_payment)
            self.energy_hist.append(allocated_energy_units_num)

            # the network(channel) state are fluctuated,
            # the transitions of which happen across the discrete decision epochs
            # following respective randomly generated matrices
            base_station_channel_state = []  # network states among all base stations at current epoch
            for base_station_index in range(BASE_STATION_NUM):
                base_station_channel_state.append(random.sample(CHANNEL_GAIN_STATE_SET, 1)[0])

            # the MU makes a joint control action for next epoch
            # noted that, in this paper, actions are two-dimension (not the same as traditional DQN)
            # hence, we need to find a map
            next_control_action = self.rl_agent.act(observation)
            next_computation_offload_decision, next_allocated_energy_units_num = \
                obtain_actual_action(CONTROL_ACTION_MAP, next_control_action)

            # record association decision in the last epoch
            last_association_decision = association_decision

            # obtain and update the base station association
            if computation_offload_decision != 0:
                # offload to a base station
                association_decision = computation_offload_decision
            elif computation_offload_decision == 0 and last_association_decision != 0:
                # current task is processed in local,
                # but the last one was processed in a certain base station
                association_decision = last_association_decision
            else:
                # for coherence
                association_decision = last_association_decision

            # store states into replay memory
            # record the next observation
            observation_raw_ = [len(task_queue), energy_queue, association_decision]
            observation_raw_.extend(base_station_channel_state)
            observation_ = np.array(observation_raw_)
            # according to computation_offload_decision and allocated_energy_units_num
            # get an one-dimension action
            self.rl_agent.remember(observation, control_action, utility, observation_)
            # update the last observation
            observation = observation_

            # if there are enough transitions, perform learning
            # at least the size of one mini_batch
            if self.rl_agent.memory_counter >= self.rl_agent.mini_batch_size:
                cur_loss = self.rl_agent.replay()
                self.loss_hist.append(cur_loss)

            # record the reward for later analysis
            self.utility_hist.append(utility)

            # update the control action for next epoch
            control_action = next_control_action
            computation_offload_decision = next_computation_offload_decision
            allocated_energy_units_num = next_allocated_energy_units_num

        return \
            self.rl_agent.model.get_weights(), \
            self.rl_agent.learning_step, \
            self.rl_agent.memory_counter, \
            self.rl_agent.memory

    def train_task(self, epoch_num, task_prob):
        # before the process begins,
        # there should be initial energy and task
        # not mentioned in paper
        task_generate = task_prob
        task_queue = []
        energy_queue = 1
        task = Task()
        task.data_size = input_data_size_each_task
        task.cpu_cycles = cpu_cycles_each_task
        task_queue.append(task)

        # begin the process simulation
        # all parameters are based on experiment1
        # assume the initial offload decision
        computation_offload_decision = 0
        allocated_energy_units_num = 1
        # obtain the 1-D action
        control_action = obtain_1_d_action(
            computation_offload_decision, allocated_energy_units_num, CONTROL_ACTION_MAP)
        association_decision = 1
        last_association_decision = association_decision
        # the network(channel) state are fluctuated,
        # the transitions of which happen across the discrete decision epochs
        # following respective randomly generated matrices
        base_station_channel_state = []  # network states among all base stations at current epoch
        for base_station_index in range(BASE_STATION_NUM):
            base_station_channel_state.append(random.sample(CHANNEL_GAIN_STATE_SET, 1)[0])

        # record the observation
        observation_raw = [len(task_queue), energy_queue, association_decision]
        observation_raw.extend(base_station_channel_state)
        observation = np.array(observation_raw)

        cur_loss = 0
        for epoch_index in range(epoch_num):
            # obtain the task execution delay
            task_execution_delay = 0
            local_time_consumption = 0
            edge_time_consumption = 0
            edge_trans_time = 0
            handover_time = 0
            # assume the execution time in the edge server is a sufficiently small constant
            edge_execution_time = 10 ** (-6)  # in second
            task_completion_num = 0
            task_fail_penalty = 0
            queuing_delay = len(task_queue)
            offload_payment = 0

            # whether the task is handled or not, the energy and the task will arrive
            # at the end of epochs, MU acquire the energy units
            # assume in this paper, the received energy units across the time horizon take integer values
            attempt_acquire_energy_units_num = scipy_poisson.rvs(ENERGY_HARVEST_LAMBDA, size=1)[0]
            # generate the task, 0 or 1
            task_arrival_indicator = scipy_bernoulli.rvs(task_generate, size=1)[0]

            # if the task queue is not empty
            # and there are enough energy units for allocation, the task could be processed
            if len(task_queue) != 0 and \
                    allocated_energy_units_num <= energy_queue:
                if allocated_energy_units_num > 0 and computation_offload_decision == 0:
                    # local computing
                    # When a computation task is scheduled for processing locally
                    # at the mobile device of the MU,
                    # calculate the allocated frequency
                    attempt_allocated_cpu_frequency = math.sqrt(
                        allocated_energy_units_num * ENERGY_UNIT /
                        (EFFECTIVE_SWITCHED_CAPACITANCE * cpu_cycles_each_task)
                    )
                    # actual consumed energy units might not equal to the allocated one
                    # due to the constrains MAX_CPU_FREQUENCY
                    allocated_cpu_frequency = 0
                    if attempt_allocated_cpu_frequency >= MAX_CPU_FREQUENCY:
                        allocated_cpu_frequency = MAX_CPU_FREQUENCY
                        allocated_energy_units_num = \
                            math.ceil(
                                math.pow(MAX_CPU_FREQUENCY, 2) *
                                EFFECTIVE_SWITCHED_CAPACITANCE * cpu_cycles_each_task / ENERGY_UNIT)
                    else:
                        allocated_cpu_frequency = attempt_allocated_cpu_frequency

                    # the time needed for local computation task execution
                    local_computation_task_execution_time = \
                        cpu_cycles_each_task / allocated_cpu_frequency
                    # for clarity
                    local_time_consumption = local_computation_task_execution_time
                    task_execution_delay = local_time_consumption
                elif allocated_energy_units_num > 0 and computation_offload_decision != 0:
                    # edge computing
                    # considering handover time
                    handover_time = calc_handover_time(
                        association_decision, last_association_decision, ONETIME_HANDOVER_DELAY)

                    # the time of transmitting task input data to edge
                    edge_trans_time, actual_energy_unit_consumption = calc_trans_time(
                        input_data_size_each_task, channel_bandwidth,
                        allocated_energy_units_num, ENERGY_UNIT,
                        base_station_channel_state[association_decision - 1],  # noted association_decision - 1
                        noise_power, MAX_TRANS_POWER)

                    # if the MAX_TRANS_POWER is being reached,
                    # obtain the actual energy unit consumption
                    if actual_energy_unit_consumption != 0:
                        allocated_energy_units_num = actual_energy_unit_consumption

                    # the time needed for a task execution in edge.
                    # include handover time, the time of transmitting task input data to edge,
                    # and edge_execution_time
                    edge_time_consumption = sum([handover_time, edge_trans_time, edge_execution_time])
                    task_execution_delay = edge_time_consumption

                    # a payment is required for the access to MEC service
                    # when the MU decides to offload a computation task for MEC server execution
                    # local computing, the payment is 0
                    mec_service_duration = min(task_execution_delay, epoch_duration) - handover_time
                    offload_payment = MEC_SERVICE_PRICE * mec_service_duration
                elif allocated_energy_units_num == 0:
                    # no energy units are allocated
                    task_execution_delay = 0
                else:
                    # in fact, the "else" is redundant
                    task_execution_delay = 0

                # to determine if the task is done
                # in this paper, only one task being handled in one epoch
                # i.e., task_completion_num = 1 or 0
                task_completion_num, queue_len_before_handling, task_fail_penalty = \
                    calc_task_completion_num(task_execution_delay, epoch_duration, task_queue)

                energy_queue = energy_queue - allocated_energy_units_num

            # update the energy queue for the next epoch
            # if the task is processed in real
            energy_queue = \
                min([energy_queue + attempt_acquire_energy_units_num,
                     MAX_ENERGY_QUEUE_LENGTH])

            # treat the queuing delay during a decision epoch j equivalently as the length of a task queue
            # if tasks are completed in this epoch, the delay should be removed
            queuing_delay = len(task_queue)

            # calculate the number of task drop
            task_drop_num = \
                max(len(task_queue) - task_completion_num + task_arrival_indicator - MAX_TASK_QUEUE_LENGTH, 0)

            # update task queue
            if task_arrival_indicator != 0 and len(task_queue) < MAX_TASK_QUEUE_LENGTH:
                # there are task arrival and queue space
                # considering future more generated tasks
                for task_arrival_index in range(task_arrival_indicator):
                    task = Task()
                    task.data_size = input_data_size_each_task
                    task.cpu_cycles = cpu_cycles_each_task
                    task_queue.append(task)
                    # MAX_TASK_QUEUE_LENGTH constrains
                    if len(task_queue) >= MAX_TASK_QUEUE_LENGTH:
                        break

            # calculate reward (utility), especially if the task (and energy) queue is not empty
            # in my opinion, when the task queue is empty, energy should not be allocated
            # however, in this paper, it should be learned
            utility = calc_utility(WEIGHTS_LIST, epoch_duration, task_execution_delay,
                                   task_drop_num, queuing_delay, task_fail_penalty, offload_payment)
            self.task_execution_delay_hist.append(task_execution_delay)
            self.task_drop_num_hist.append(task_drop_num)
            self.queuing_delay_hist.append(queuing_delay)
            self.task_fail_penalty_hist.append(task_fail_penalty)
            self.offload_payment_hist.append(offload_payment)
            self.energy_hist.append(allocated_energy_units_num)

            # the network(channel) state are fluctuated,
            # the transitions of which happen across the discrete decision epochs
            # following respective randomly generated matrices
            base_station_channel_state = []  # network states among all base stations at current epoch
            for base_station_index in range(BASE_STATION_NUM):
                base_station_channel_state.append(random.sample(CHANNEL_GAIN_STATE_SET, 1)[0])

            # the MU makes a joint control action for next epoch
            # noted that, in this paper, actions are two-dimension (not the same as traditional DQN)
            # hence, we need to find a map
            next_control_action = self.rl_agent.act(observation)
            next_computation_offload_decision, next_allocated_energy_units_num = \
                obtain_actual_action(CONTROL_ACTION_MAP, next_control_action)

            # record association decision in the last epoch
            last_association_decision = association_decision

            # obtain and update the base station association
            if computation_offload_decision != 0:
                # offload to a base station
                association_decision = computation_offload_decision
            elif computation_offload_decision == 0 and last_association_decision != 0:
                # current task is processed in local,
                # but the last one was processed in a certain base station
                association_decision = last_association_decision
            else:
                # for coherence
                association_decision = last_association_decision

            # store states into replay memory
            # record the next observation
            observation_raw_ = [len(task_queue), energy_queue, association_decision]
            observation_raw_.extend(base_station_channel_state)
            observation_ = np.array(observation_raw_)
            # according to computation_offload_decision and allocated_energy_units_num
            # get an one-dimension action
            self.rl_agent.remember(observation, control_action, utility, observation_)
            # update the last observation
            observation = observation_

            # if there are enough transitions, perform learning
            # at least the size of one mini_batch
            if self.rl_agent.memory_counter >= self.rl_agent.mini_batch_size:
                cur_loss = self.rl_agent.replay()
                self.loss_hist.append(cur_loss)

            # record the reward for later analysis
            self.utility_hist.append(utility)

            # update the control action for next epoch
            control_action = next_control_action
            computation_offload_decision = next_computation_offload_decision
            allocated_energy_units_num = next_allocated_energy_units_num

        return \
            self.rl_agent.model.get_weights(), \
            self.rl_agent.learning_step, \
            self.rl_agent.memory_counter, \
            self.rl_agent.memory

    def train_batch(self, epoch_num, batch):
        self.rl_agent.mini_batch_size = batch
        task_generate_prob = 0.5
        # before the process begins,
        # there should be initial energy and task
        # not mentioned in paper
        task_queue = []
        energy_queue = 1
        task = Task()
        task.data_size = input_data_size_each_task
        task.cpu_cycles = cpu_cycles_each_task
        task_queue.append(task)

        # begin the process simulation
        # all parameters are based on experiment1
        # assume the initial offload decision
        computation_offload_decision = 0
        allocated_energy_units_num = 1
        # obtain the 1-D action
        control_action = obtain_1_d_action(
            computation_offload_decision, allocated_energy_units_num, CONTROL_ACTION_MAP)
        association_decision = 1
        last_association_decision = association_decision
        # the network(channel) state are fluctuated,
        # the transitions of which happen across the discrete decision epochs
        # following respective randomly generated matrices
        base_station_channel_state = []  # network states among all base stations at current epoch
        for base_station_index in range(BASE_STATION_NUM):
            base_station_channel_state.append(random.sample(CHANNEL_GAIN_STATE_SET, 1)[0])

        # record the observation
        observation_raw = [len(task_queue), energy_queue, association_decision]
        observation_raw.extend(base_station_channel_state)
        observation = np.array(observation_raw)

        cur_loss = 0
        for epoch_index in range(epoch_num):
            # obtain the task execution delay
            task_execution_delay = 0
            local_time_consumption = 0
            edge_time_consumption = 0
            edge_trans_time = 0
            handover_time = 0
            # assume the execution time in the edge server is a sufficiently small constant
            edge_execution_time = 10 ** (-6)  # in second
            task_completion_num = 0
            task_fail_penalty = 0
            queuing_delay = len(task_queue)
            offload_payment = 0

            # whether the task is handled or not, the energy and the task will arrive
            # at the end of epochs, MU acquire the energy units
            # assume in this paper, the received energy units across the time horizon take integer values
            attempt_acquire_energy_units_num = scipy_poisson.rvs(ENERGY_HARVEST_LAMBDA, size=1)[0]
            # generate the task, 0 or 1
            task_arrival_indicator = scipy_bernoulli.rvs(task_generate_prob, size=1)[0]

            # if the task queue is not empty
            # and there are enough energy units for allocation, the task could be processed
            if len(task_queue) != 0 and \
                    allocated_energy_units_num <= energy_queue:
                if allocated_energy_units_num > 0 and computation_offload_decision == 0:
                    # local computing
                    # When a computation task is scheduled for processing locally
                    # at the mobile device of the MU,
                    # calculate the allocated frequency
                    attempt_allocated_cpu_frequency = math.sqrt(
                        allocated_energy_units_num * ENERGY_UNIT /
                        (EFFECTIVE_SWITCHED_CAPACITANCE * cpu_cycles_each_task)
                    )
                    # actual consumed energy units might not equal to the allocated one
                    # due to the constrains MAX_CPU_FREQUENCY
                    allocated_cpu_frequency = 0
                    if attempt_allocated_cpu_frequency >= MAX_CPU_FREQUENCY:
                        allocated_cpu_frequency = MAX_CPU_FREQUENCY
                        allocated_energy_units_num = \
                            math.ceil(
                                math.pow(MAX_CPU_FREQUENCY, 2) *
                                EFFECTIVE_SWITCHED_CAPACITANCE * cpu_cycles_each_task / ENERGY_UNIT)
                    else:
                        allocated_cpu_frequency = attempt_allocated_cpu_frequency

                    # the time needed for local computation task execution
                    local_computation_task_execution_time = \
                        cpu_cycles_each_task / allocated_cpu_frequency
                    # for clarity
                    local_time_consumption = local_computation_task_execution_time
                    task_execution_delay = local_time_consumption
                elif allocated_energy_units_num > 0 and computation_offload_decision != 0:
                    # edge computing
                    # considering handover time
                    handover_time = calc_handover_time(
                        association_decision, last_association_decision, ONETIME_HANDOVER_DELAY)

                    # the time of transmitting task input data to edge
                    edge_trans_time, actual_energy_unit_consumption = calc_trans_time(
                        input_data_size_each_task, channel_bandwidth,
                        allocated_energy_units_num, ENERGY_UNIT,
                        base_station_channel_state[association_decision - 1],  # noted association_decision - 1
                        noise_power, MAX_TRANS_POWER)

                    # if the MAX_TRANS_POWER is being reached,
                    # obtain the actual energy unit consumption
                    if actual_energy_unit_consumption != 0:
                        allocated_energy_units_num = actual_energy_unit_consumption

                    # the time needed for a task execution in edge.
                    # include handover time, the time of transmitting task input data to edge,
                    # and edge_execution_time
                    edge_time_consumption = sum([handover_time, edge_trans_time, edge_execution_time])
                    task_execution_delay = edge_time_consumption

                    # a payment is required for the access to MEC service
                    # when the MU decides to offload a computation task for MEC server execution
                    # local computing, the payment is 0
                    mec_service_duration = min(task_execution_delay, epoch_duration) - handover_time
                    offload_payment = MEC_SERVICE_PRICE * mec_service_duration
                elif allocated_energy_units_num == 0:
                    # no energy units are allocated
                    task_execution_delay = 0
                else:
                    # in fact, the "else" is redundant
                    task_execution_delay = 0

                # to determine if the task is done
                # in this paper, only one task being handled in one epoch
                # i.e., task_completion_num = 1 or 0
                task_completion_num, queue_len_before_handling, task_fail_penalty = \
                    calc_task_completion_num(task_execution_delay, epoch_duration, task_queue)

                energy_queue = energy_queue - allocated_energy_units_num

            # update the energy queue for the next epoch
            # if the task is processed in real
            energy_queue = \
                min([energy_queue + attempt_acquire_energy_units_num,
                     MAX_ENERGY_QUEUE_LENGTH])

            # treat the queuing delay during a decision epoch j equivalently as the length of a task queue
            # if tasks are completed in this epoch, the delay should be removed
            queuing_delay = len(task_queue)

            # calculate the number of task drop
            task_drop_num = \
                max(len(task_queue) - task_completion_num + task_arrival_indicator - MAX_TASK_QUEUE_LENGTH, 0)

            # update task queue
            if task_arrival_indicator != 0 and len(task_queue) < MAX_TASK_QUEUE_LENGTH:
                # there are task arrival and queue space
                # considering future more generated tasks
                for task_arrival_index in range(task_arrival_indicator):
                    task = Task()
                    task.data_size = input_data_size_each_task
                    task.cpu_cycles = cpu_cycles_each_task
                    task_queue.append(task)
                    # MAX_TASK_QUEUE_LENGTH constrains
                    if len(task_queue) >= MAX_TASK_QUEUE_LENGTH:
                        break

            # calculate reward (utility), especially if the task (and energy) queue is not empty
            # in my opinion, when the task queue is empty, energy should not be allocated
            # however, in this paper, it should be learned
            utility = calc_utility(WEIGHTS_LIST, epoch_duration, task_execution_delay,
                                   task_drop_num, queuing_delay, task_fail_penalty, offload_payment)
            self.task_execution_delay_hist.append(task_execution_delay)
            self.task_drop_num_hist.append(task_drop_num)
            self.queuing_delay_hist.append(queuing_delay)
            self.task_fail_penalty_hist.append(task_fail_penalty)
            self.offload_payment_hist.append(offload_payment)
            self.energy_hist.append(allocated_energy_units_num)

            # the network(channel) state are fluctuated,
            # the transitions of which happen across the discrete decision epochs
            # following respective randomly generated matrices
            base_station_channel_state = []  # network states among all base stations at current epoch
            for base_station_index in range(BASE_STATION_NUM):
                base_station_channel_state.append(random.sample(CHANNEL_GAIN_STATE_SET, 1)[0])

            # the MU makes a joint control action for next epoch
            # noted that, in this paper, actions are two-dimension (not the same as traditional DQN)
            # hence, we need to find a map
            next_control_action = self.rl_agent.act(observation)
            next_computation_offload_decision, next_allocated_energy_units_num = \
                obtain_actual_action(CONTROL_ACTION_MAP, next_control_action)

            # record association decision in the last epoch
            last_association_decision = association_decision

            # obtain and update the base station association
            if computation_offload_decision != 0:
                # offload to a base station
                association_decision = computation_offload_decision
            elif computation_offload_decision == 0 and last_association_decision != 0:
                # current task is processed in local,
                # but the last one was processed in a certain base station
                association_decision = last_association_decision
            else:
                # for coherence
                association_decision = last_association_decision

            # store states into replay memory
            # record the next observation
            observation_raw_ = [len(task_queue), energy_queue, association_decision]
            observation_raw_.extend(base_station_channel_state)
            observation_ = np.array(observation_raw_)
            # according to computation_offload_decision and allocated_energy_units_num
            # get an one-dimension action
            self.rl_agent.remember(observation, control_action, utility, observation_)
            # update the last observation
            observation = observation_

            # if there are enough transitions, perform learning
            # at least the size of one mini_batch
            if self.rl_agent.memory_counter >= self.rl_agent.mini_batch_size:
                cur_loss = self.rl_agent.replay()
                self.loss_hist.append(cur_loss)

            # record the reward for later analysis
            self.utility_hist.append(utility)

            # update the control action for next epoch
            control_action = next_control_action
            computation_offload_decision = next_computation_offload_decision
            allocated_energy_units_num = next_allocated_energy_units_num

        return \
            self.rl_agent.model.get_weights(), \
            self.rl_agent.learning_step, \
            self.rl_agent.memory_counter, \
            self.rl_agent.memory
