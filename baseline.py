import os
import pickle
import numpy as np
from pykalman import KalmanFilter
from utils import displaybbox

def use_baseline(script_args, hype, MASTER_ROOT_DIR):
    for index in range(script_args['DATA_AMOUNT_EVAL_START'], script_args['DATA_AMOUNT_EVAL_END']):
        start_index = index*100
        end_index = start_index + 100
        print("NOW LOADING IMAGES FROM " + str(start_index) + " TO " + str(end_index))

        fp = open(os.path.join(MASTER_ROOT_DIR, script_args['ROOT_DIR'], "bbox_files", "bboxes" + str(start_index) +
                               "to" + str(end_index)) + ".txt", "rb")
        list_of_dicts = pickle.load(fp)
        total_seq_len = hype['input_seq_len'] + hype['target_seq_len']
        for j in range((end_index - start_index) // total_seq_len):
            start_img = j * total_seq_len
            end_img = start_img + total_seq_len

            all_agents_subset = []
            for dict in list_of_dicts[start_img:end_img]:
                for agent in dict.keys():
                    all_agents_subset.append(agent)
            all_agents_subset_unique = list(set(all_agents_subset))
            print("TOTAL UNIQUE AGENTS " + str(len(all_agents_subset_unique)))
            for current_agent in all_agents_subset_unique:
                current_agent_counter = 0
                for dict in list_of_dicts[start_img:end_img]:
                    if current_agent in dict.keys():
                        current_agent_counter += 1
                if current_agent_counter == total_seq_len:
                    print("NOW WORKING ON AGENT " + str(current_agent))

                    if script_args['BASELINE'] in ['AVG']:
                        avg_bbox_diff = np.zeros(4)
                        agent_bbox = list_of_dicts[start_img][current_agent]
                        for dict in list_of_dicts[start_img:start_img + script_args['input_seq_len']]:
                            avg_bbox_diff += (dict[current_agent] - agent_bbox)
                        last_known_agent_bbox = list_of_dicts[start_img + script_args['input_seq_len'] - 1][current_agent]
                        prediction = last_known_agent_bbox + script_args['target_seq_len'] * \
                                     (avg_bbox_diff / (script_args['input_seq_len'] - 1))

                        for dict in list_of_dicts[start_img:end_img]:
                            dict[str(current_agent) + ' (PREDICTION)'] = prediction
                    elif script_args['BASELINE'] in ['KALMAN']:
                        measurements = []
                        for dict in list_of_dicts[start_img:start_img + script_args['input_seq_len']]:
                            measurements.append(dict[current_agent])
                        measurements_np = np.asarray(measurements)
                        initial_state_mean = [measurements_np[0, 0],
                                              0,
                                              measurements_np[0, 1],
                                              0,
                                              measurements_np[0, 2],
                                              0,
                                              measurements_np[0, 3],
                                              0]
                        transition_matrix = [[1, 1, 0, 0, 0, 0, 0, 0],
                                             [0, 1, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 1, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 1, 0, 0],
                                             [0, 0, 0, 0, 0, 1, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 1, 1],
                                             [0, 0, 0, 0, 0, 0, 0, 1]]
                        observation_matrix = [[1, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 1, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 1, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 1, 0]]
                        kf = KalmanFilter(transition_matrices=transition_matrix,
                                           observation_matrices=observation_matrix,
                                           initial_state_mean=initial_state_mean)

                        for pred in range(script_args['target_seq_len']):
                            kf = kf.em(measurements_np, n_iter=5)
                            kf.smooth(measurements_np)
                        prediction = np.array([measurements_np[-1,0], measurements_np[-1,1], measurements_np[-1,2],
                                               measurements_np[-1,3]])
                        for dict in list_of_dicts[start_img:end_img]:
                            dict[str(current_agent) + ' (PREDICTION)'] = prediction
                else:
                    print("AGENT " + str(current_agent) + " NOT FULLY PRESENT")

        with open("predictions_baseline" + str(start_index) + "to" + str(end_index) + ".txt", "wb") as fp:
            pickle.dump(list_of_dicts, fp)

def show_predictions(script_args, data, MASTER_ROOT_DIR, fileName):
    for index in range(script_args['DATA_AMOUNT_EVAL_START'], script_args['DATA_AMOUNT_EVAL_END']):
        start_index = index*100
        end_index = start_index + 100
        print("NOW SHOWING PREDICTIONS FOR IMAGES FROM " + str(start_index) + " TO " + str(end_index))
        fp = open(fileName + str(start_index) + "to" + str(end_index) + ".txt", "rb")

        list_of_dicts = pickle.load(fp)

        displaybbox(list_of_dicts, data, start_index, end_index, os.path.join(MASTER_ROOT_DIR, script_args['ROOT_DIR'],
                                                                              script_args['TRAIN_DIR_UNMASKED']))