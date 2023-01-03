# rl: reinforcement learning
import json
import tempfile
from DLAgent import DQNAgent
from Server import Server
from Client import Client
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_model_optimization as tfmot



# 按segment_length分割列表成子列表，并计算分别的平均数
def calc_segment_mean(utility_hist, segment_length):
    segment_mean_list = []
    max_segment_num = int(len(utility_hist) / segment_length)
    utility_array = np.array_split(np.array(utility_hist), max_segment_num)
    for idx in range(len(utility_array)):
        segment_mean_list.append(np.mean(utility_array[idx]))
    return segment_mean_list


def fl_dqn_offload(task_pro_fn, agent_num_fn, epoch_num_fn):
    task_generate_prob = task_pro_fn
    # 共用参数设定
    # computing offload model
    BASE_STATION_NUM = 6
    MAX_ENERGY_QUEUE_LENGTH = 4  # in units

    # 云端初始化一个RL模型
    server = Server()
    # initiate DeepQNetwork
    # Rl parameters
    DISCOUNT_FACTOR = 0.9
    EXPLORATION_PROBABILITY = 0.01
    REPLAY_MEMORY_CAPACITY = 5000
    MINI_BATCH_SIZE = 200  # 200
    REPLACE_TARGET_PERIOD = 250
    # include the computation offloading decision
    # and the number of allocated energy units
    # initiation
    # need this map to find action
    # 第一个+1是什么？ 第二个+1是不发吗
    flat_action_num = (BASE_STATION_NUM + 1) * (MAX_ENERGY_QUEUE_LENGTH + 1)
    observation_dimension = 1 + 1 + 1 + BASE_STATION_NUM

    global_rl_agent = DQNAgent(observation_dimension, flat_action_num,
                               gamma=DISCOUNT_FACTOR,
                               replay_memory_size=REPLAY_MEMORY_CAPACITY,
                               mini_batch_size=MINI_BATCH_SIZE,
                               replace_target_period=REPLACE_TARGET_PERIOD,
                               epsilon_min=EXPLORATION_PROBABILITY,
                               )

    # 存下global_rl_agent中的神经网络模型
    global_model_weight = global_rl_agent.model.get_weights()

    simulate_times = 100  # 过程模拟次数
    client_num = agent_num_fn  # 模拟的Client数目
    client_list = []

    # 初始化所有Client
    for client_index in range(client_num):
        client_list.append(
            Client(global_rl_agent, global_model_weight, client_id=client_index))

    # 贯穿整个模拟过程
    # 存下每个Client训练后的learning_step
    client_learning_step_list = [0 for client_index in range(client_num)]
    # 存下每个Client训练后的memory_counter
    client_memory_counter_list = [0 for client_index in range(client_num)]
    # 存下每个Client训练后的memory
    client_memory_list = [np.zeros_like(global_rl_agent.memory)
                          for client_index in range(client_num)]

    for simulate_index in range(simulate_times):
        # 将初始的神经网络模型下发到各个Client进行分布式训练
        # 这里假设每个client训练(epoch_num)次
        epoch_num = epoch_num_fn

        client_weight_update_list = []  # 存下每个Client训练后的模型权重
        client_train_size_list = []  # 存下每个Client训练数据的大小

        # 模拟多个Client来分布式训练
        # 模拟单个Client分别训练，分别训练的次数为(epoch_num - mini_batch_size + 1)
        # if_prue = False
        for client_index in range(client_num):
            print("进度：", simulate_index * client_num + client_index + 1, "/", simulate_times * client_num)
            client = client_list[client_index]
            client.update_model_from_server(global_model_weight)
            # 恢复所有Client的原本状态
            client.rl_agent.learning_step = client_learning_step_list[client.ID]
            client.rl_agent.memory_counter = client_memory_counter_list[client.ID]
            client.rl_agent.memory = client_memory_list[client.ID]

            # if client_index == client_num-1:
            #     if_prue = True
            # 模拟替换掉不同Client的独立部分
            client_weights, client_learning_step, client_memory_counter, client_memory = \
                client.train(epoch_num, task_generate_prob)
            # for merging model
            # 记录每个Client训练后的模型权重更新
            client_weight_update_list.append(client_weights)
            # 记录每个Client训练的次数
            # client_train_size_list.append(epoch_num - MINI_BATCH_SIZE + 1)
            client_train_size_list.append(
                client.rl_agent.learning_step - client_learning_step_list[client.ID] + 1)
            # for restore state of client
            client_learning_step_list[client.ID] = client_learning_step
            client_memory_counter_list[client.ID] = client_memory_counter
            client_memory_list[client.ID] = client_memory
            # 为了模拟不同Client之间的独立，清空global_rl_agent的状态
            global_rl_agent.model.set_weights(global_model_weight)
            global_rl_agent.memory = []
            global_rl_agent.learning_step = 0
            global_rl_agent.memory_counter = 0

        # 云端综合处理所有Client模型的更新，
        new_weights = [np.zeros(weight.shape)
                       for weight in global_rl_agent.model.get_weights()]
        total_train_size = np.sum(client_train_size_list)
        for client_index in range(len(client_weight_update_list)):
            for weight_index in range(len(new_weights)):
                new_weights[weight_index] += \
                    client_weight_update_list[client_index][weight_index] * \
                    client_train_size_list[client_index] / total_train_size

        # 云端汇聚model进行总体更新
        # 并对global_model_weight更新后再分发给所有的Client
        global_model_weight = new_weights
        global_rl_agent.model.set_weights(global_model_weight)

    # 计算经验性判断神经网络收敛之后 所有Client的平均Reward (Utility)
    client_mean_utility_list_after_convergence = []
    convergence_index = 6 * 10 ** 3
    # for client_index in range(client_num):
    #     client = client_list[client_index]
    #     client_mean_utility_list_after_convergence.append(
    #         np.mean(client.utility_hist[convergence_index:]))

    # task_execution_delay,task_drop_num, queuing_delay, task_fail_penalty, offload_payment

    client = client_list[0]
    client_task_execution_delay = client.task_execution_delay_hist
    client_task_drop_num = client.task_drop_num_hist
    client_queuing_delay = client.queuing_delay_hist
    client_task_fail_penalty = client.task_fail_penalty_hist
    client_offload_payment = client.offload_payment_hist
    client_mean_utility_list_after_convergence = client.utility_hist
    energy_list = client.energy_hist
    for utility_index in range(len(client_mean_utility_list_after_convergence)):
        for client_index in range(client_num):
            client = client_list[client_index]
            if client_index == 0:
                pass
            else:
                client_mean_utility_list_after_convergence[utility_index] = (float)(
                    client_mean_utility_list_after_convergence[utility_index] + client.utility_hist[utility_index])
                client_task_execution_delay[utility_index] = (float)(
                    client_task_execution_delay[utility_index] + client.task_execution_delay_hist[utility_index])
                client_task_drop_num[utility_index] = (float)(client_task_drop_num[utility_index] + \
                                                              client.task_drop_num_hist[utility_index])
                client_queuing_delay[utility_index] = (float)(client_queuing_delay[utility_index] + \
                                                              client.queuing_delay_hist[utility_index])
                client_task_fail_penalty[utility_index] = (float)(client_task_fail_penalty[utility_index] + \
                                                                  client.task_fail_penalty_hist[utility_index])
                client_offload_payment[utility_index] = (float)(client_offload_payment[utility_index] + \
                                                                client.offload_payment_hist[utility_index])
                # print("utility_index", utility_index, "client.energy_hist[utility_index]", len(client.energy_hist),
                #       len(client.offload_payment_hist))
                energy_list[utility_index] = (float)(energy_list[utility_index] + \
                                                     client.energy_hist[utility_index])

        client_mean_utility_list_after_convergence[utility_index] = client_mean_utility_list_after_convergence[
                                                                        utility_index] / client_num
        client_task_execution_delay[utility_index] = client_task_execution_delay[utility_index] / client_num
        client_task_drop_num[utility_index] = client_task_drop_num[utility_index] / client_num
        client_queuing_delay[utility_index] = client_queuing_delay[utility_index] / client_num
        client_task_fail_penalty[utility_index] = client_task_fail_penalty[utility_index] / client_num
        client_offload_payment[utility_index] = client_offload_payment[utility_index] / client_num
        energy_list[utility_index] = energy_list[utility_index] / client_num
    return client_mean_utility_list_after_convergence, client_task_execution_delay, client_task_drop_num, client_queuing_delay, client_task_fail_penalty, client_offload_payment, energy_list, global_rl_agent


def get_gzipped_model_size(file):
    # Returns size of gzipped model, in bytes.
    import os
    import zipfile

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)

    return os.path.getsize(zipped_file)


if __name__ == '__main__':
    prue_num = '0.6'
    print(prue_num)

    epoch_num = 100
    task_pro = 0.5
    agent_num = 7
    total_num = 1

    utility = {}
    task_execution_delay = {}
    task_drop_num = {}
    queuing_delay = {}
    task_fail_penalty = {}
    offload_payment = {}
    energy_list = {}
    for total_fn in range(total_num):
        print("总进度：", total_fn, "/", total_num)
        print("总进度：", total_fn, "/", total_num)
        print("总进度：", total_fn, "/", total_num)
        print("总进度：", total_fn, "/", total_num)
        print("总进度：", total_fn, "/", total_num)
        print("总进度：", total_fn, "/", total_num)
        # utility,task_execution_delay,task_drop_num, queuing_delay, task_fail_penalty, offload_payment
        utility[total_fn], task_execution_delay[total_fn], task_drop_num[total_fn], queuing_delay[total_fn], \
        task_fail_penalty[total_fn], offload_payment[total_fn], energy_list[total_fn], global_rl_agent = fl_dqn_offload(
            task_pro,
            agent_num,
            epoch_num)

    global_rl_agent.target_model.summary()

    pruned_keras_file = prue_num + 'pruned_keras_file_7agent.h5'
    tf.keras.models.save_model(global_rl_agent.target_model, pruned_keras_file, include_optimizer=False)
    print('Saved pruned Keras model to:', pruned_keras_file)

    model_for_export = tfmot.sparsity.keras.strip_pruning(global_rl_agent.target_model)

    converter = tf.lite.TFLiteConverter.from_keras_model_file(model_for_export)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    pruned_tflite_model = converter.convert()

    pruned_tflite_file = prue_num + 'pruned_keras_file_5agent.tflite'

    with open(pruned_tflite_file, 'wb') as f:
        f.write(pruned_tflite_model)

    print('Saved quantized and pruned TFLite model to:', pruned_tflite_file)

    print("Size of gzipped pruned TFlite model: %.2f bytes" % (
        get_gzipped_model_size(pruned_tflite_file)))

    file_object = open('./' + prue_num + '_7agent.txt', 'w')
    file_object.write(str(get_gzipped_model_size(pruned_tflite_file)))
    file_object.close()

    file_object = open('./' + prue_num + '_7agent_try_utility.txt', 'w')
    file_object.write(str(utility))
    file_object.close()
    plt.plot(range(len(utility[0])), utility[0])
    plt.ylabel('Utility')
    plt.xlabel('Training Period')
    plt.show()

    file_object = open('./' + prue_num + '_7agent_try_task_execution_delay.txt', 'w')
    file_object.write(str(task_execution_delay))
    file_object.close()
    plt.plot(range(len(task_execution_delay[0])), task_execution_delay[0])
    plt.ylabel('Task Execution Delay')
    plt.xlabel('Training Period')
    plt.show()

    file_object = open('./' + prue_num + '_7agent_try_task_drop_num.txt', 'w')
    file_object.write(str(task_drop_num))
    file_object.close()
    plt.plot(range(len(task_drop_num[0])), task_drop_num[0])
    plt.ylabel('Task Drop Num')
    plt.xlabel('Training Period')
    plt.show()

    file_object = open('./' + prue_num + '_7agent_try_queuing_delay.txt', 'w')
    file_object.write(str(queuing_delay))
    file_object.close()
    plt.plot(range(len(queuing_delay[0])), queuing_delay[0])
    plt.ylabel('Queuing Delay')
    plt.xlabel('Training Period')
    plt.show()

    file_object = open('./' + prue_num + '_7agent_try_task_fail_penalty.txt', 'w')
    file_object.write(str(task_fail_penalty))
    file_object.close()
    plt.plot(range(len(task_fail_penalty[0])), task_fail_penalty[0])
    plt.ylabel('Task Fail Penalty')
    plt.xlabel('Training Period')
    plt.show()

    file_object = open('./' + prue_num + '_7agent_try_offload_payment.txt', 'w')
    file_object.write(str(offload_payment))
    file_object.close()
    plt.plot(range(len(offload_payment[0])), offload_payment[0])
    plt.ylabel('Offload Payment')
    plt.xlabel('Training Period')
    plt.show()

    file_object = open('./' + prue_num + '_7agent_try_energy.txt', 'w')
    file_object.write(str(energy_list))
    file_object.close()
    plt.plot(range(len(energy_list[0])), energy_list[0])
    plt.ylabel('energy')
    plt.xlabel('Training Period')
    plt.show()
    print('prue_num', prue_num)