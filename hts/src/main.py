import math
import numpy as np

from experiment import Experiment

EX = Experiment()

task_num = EX.get_task_num()

for task_idx in range(task_num):
    data = EX.get_data(task_idx+1)

    exp_list = []
    rank_list = []
    value_list = []

    for roop_i in range(500):
        current_exp = math.floor(5000*np.random.rand())

        results_dict = EX.exp(task_id=task_idx+1, chem_id=current_exp)
        exp_list = np.append(exp_list, current_exp)
        rank_list = np.append(rank_list, results_dict["rank"])
        value_list = np.append(value_list, results_dict["value"])
