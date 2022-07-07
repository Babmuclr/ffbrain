import pandas as pd
import os
import numpy as np
import time
import datetime
import glob
import json

class Experiment:
    def __init__(self):
        self.__rank_criterion = 50
        self.__max_iteration = 500

        DIR = os.path.dirname(__file__)
        if len(DIR) < 1:
            self.__DIR = "./"
        else:
            self.__DIR = DIR+"/"
        
        self.__dataset_files_path = sorted(glob.glob(self.__DIR + "../sample/*"))
        self.__task_num = len(self.__dataset_files_path)
        
        self.__all_private_df = pd.DataFrame()
        for f in self.__dataset_files_path:
            self.__tmp_df = pd.read_csv(f)
            self.__all_private_df = pd.concat([self.__all_private_df, self.__tmp_df])
        
        self.__all_private_df = self.__all_private_df.reset_index(drop=True)

        self.__iteration_list = [0] * self.__task_num
        self.__countup_score_list = [0] * self.__task_num
        self.__biased_score_list = [0] * self.__task_num
        self.__chem_id_list = [[] for _ in range(self.__task_num)]
        self.__start_time = time.time()

    def get_task_num(self):
        return self.__task_num

    def exp(self, task_id, chem_id):
        value = self.__all_private_df.iloc[(task_id-1)*5000+chem_id, 1]
        rank = self.__all_private_df.iloc[(task_id-1)*5000+chem_id, 2]

        results_dict = {}
        results_dict["value"] = value
        results_dict["rank"] = rank

        if self.__iteration_list[task_id-1] < self.__max_iteration:

            if (rank <= self.__rank_criterion) and (chem_id not in self.__chem_id_list[task_id-1]):
                self.__countup_score_list[task_id-1] = self.__countup_score_list[task_id-1] + 1
                self.__biased_score_list[task_id-1] = self.__biased_score_list[task_id-1] + (self.__rank_criterion - int(rank) + 1)

            self.__chem_id_list[task_id-1].append(chem_id)
            self.__iteration_list[task_id-1] = self.__iteration_list[task_id-1] + 1
            elapsed_time = time.time() - self.__start_time
            elapsed_datetime = datetime.timedelta(seconds=elapsed_time)
            if (self.__iteration_list[task_id-1] == self.__max_iteration):
                print(f"final countup_score = {self.__countup_score_list[task_id-1]},  elapsed_time = {elapsed_datetime}")

            # すべてのタスクで500回実験が終わったら、各スコアの平均を表示
            if (len(set(self.__iteration_list))==1) and (list(set(self.__iteration_list))[0] == self.__max_iteration):
                print("###############")
                print(f"average countup_score = {np.mean(self.__countup_score_list)},  elapsed_time = {elapsed_datetime}")

        return results_dict
        
    def get_data(self, task_id):
        target_task_df = self.__all_private_df.drop(['Potency', 'rank'], axis=1, errors='ignore')[(task_id-1)*5000:task_id*5000].reset_index(drop=True)

        return target_task_df