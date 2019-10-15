import cv2
import torch
import os
import csv
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset

def displaybbox(list_of_dicts, json_data, starting_index, ending_index, img_dir):
    size = (0, 0)
    matImgArray = []

    print("LENGTH OF LIST " + str(len(list_of_dicts)))
    img_array = []
    for frame in json_data[starting_index:ending_index]:
        if frame['labels'] is not None:
            img_array.append(frame['name'])
    print("LENGTH OF IMAGE ARRAY " + str(len(img_array)))

    current_frame = 0
    for img in img_array:
        matImg = cv2.imread(os.path.join(img_dir, img))
        frame_dict = list_of_dicts[current_frame]
        for agent_id, bbox in frame_dict.items():
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[2]), int(bbox[3]))

            if 'PREDICTION' not in str(agent_id):
                cv2.rectangle(matImg, p1, p2, (255, 255, 255), 2, 1)
                cv2.putText(matImg, "AGENT " + str(agent_id), (p1[0] - 10, p1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2,
                            cv2.LINE_AA)
            else:
                cv2.rectangle(matImg, p1, p2, (255, 255, 0), 2, 1)
                cv2.putText(matImg, "AGENT " + str(agent_id), (p1[0] - 10, p1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2,
                            cv2.LINE_AA)
        height, width, layers = matImg.shape
        size = (width, height)
        matImgArray.append(matImg)
        current_frame += 1

    print("NOW CREATING VIDEO")
    out = cv2.VideoWriter("PROJECT" + str(starting_index) + " TO " + str(ending_index) + ".avi",
                          cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for j in range(len(matImgArray)):
        out.write(matImgArray[j])
    out.release()

def convertToEncoderInput(tensorToConvert, seq_len, batch_size):
    offset = 1
    newTensor = tensorToConvert[0:seq_len]
    for i in range(batch_size - 1):
        newTensor = torch.cat((newTensor, tensorToConvert[offset:offset + seq_len]), dim=0)
        offset += 1
    newTensor = newTensor.unsqueeze(-1)
    newTensor = newTensor.view(seq_len, batch_size, newTensor.size()[1])

    return newTensor

def flip_bbox(width, bbox):
    new_bbox = np.zeros(4)
    new_bbox[0] = width + (width - bbox[0]) - abs(bbox[0] - bbox[2])
    new_bbox[1] = bbox[1]
    new_bbox[2] = width + (width - bbox[2]) + abs(bbox[0] - bbox[2])
    new_bbox[3] = bbox[3]

    return new_bbox

def convert_to_dict(filepath):
    f = open(filepath, 'r')
    reader = csv.reader(f)
    your_dict = {}
    for row in reader:
        if "list_int" in row[-1]:
            your_dict[row[0]] = []
            for element in row[1:-1]:
                your_dict[row[0]].append(int(element))
        elif "int" in row[-1]:
            your_dict[row[0]] = int(row[1])
        elif "float" in row[-1]:
            your_dict[row[0]] = float(row[1])
        elif "bool" in row[-1] and "True" in row[1]:
            your_dict[row[0]] = True
        elif "bool" in row[-1] and "False" in row[1]:
            your_dict[row[0]] = False
        elif "string" in row[-1]:
            your_dict[row[0]] = row[1]
        elif "list_string" in row[-1]:
            your_dict[row[0]] = []
            for element in row[1:-1]:
                your_dict[row[0]].append(element)
    f.close()
    return your_dict

def create_data(index, trainFile, DATASET, indices, script_args, data, batch_size, MASTER_ROOT_DIR):
    start_index = index * 100
    end_index = start_index + 100
    print("NOW LOADING IMAGES FROM " + str(start_index) + " TO " + str(end_index))
    trainFile.write("NOW LOADING IMAGES FROM " + str(start_index) + " TO " + str(end_index) + "\n")
    _DATASET = Subset(DATASET, indices[start_index:end_index])

    masked_img_array = []
    masked_img_flipped_array = []
    unmasked_img_array = []
    loader = DataLoader(dataset=_DATASET, batch_size=batch_size, shuffle=False, num_workers=script_args['num_workers'])

    for i, (image_batch, image_batch_flipped) in enumerate(loader):
        masked_img_array.append(image_batch)
        masked_img_flipped_array.append(image_batch_flipped)

    for frame in data[start_index:end_index]:
        if frame['labels'] is not None:
            unmasked_img_array.append(frame['name'])

    fp = open(os.path.join(MASTER_ROOT_DIR, script_args['ROOT_DIR'], "bbox_files", "bboxes" + str(start_index) +
                           "to" + str(end_index)) + ".txt", "rb")
    list_of_dicts = pickle.load(fp)
    all_agents = []
    for dict in list_of_dicts:
        for agent in dict.keys():
            all_agents.append(agent)

    all_agents_unique = list(set(all_agents))
    print("TOTAL UNIQUE AGENTS " + str(len(all_agents_unique)))
    trainFile.write("TOTAL UNIQUE AGENTS " + str(len(all_agents_unique)) + "\n")

    return all_agents_unique, list_of_dicts, masked_img_array, masked_img_flipped_array, unmasked_img_array

def feature_extraction(dict_of_models, device, hype, input):
    mask_conv_output = dict_of_models['MASKED CONV MODEL'](input[0].to(device))
    mask_conv_output = torch.cat((mask_conv_output, input[1]), 1)
    agent_conv_output = dict_of_models['UNMASKED CONV MODEL'](input[4].to(device))
    final_mask_conv_output = torch.cat((mask_conv_output, agent_conv_output), 1)
    final_mask_conv_features = convertToEncoderInput(final_mask_conv_output, hype['input_seq_len'], hype['batch_size'])
    final_target_features = convertToEncoderInput(input[3], hype['target_seq_len'], hype['batch_size'])
    return final_mask_conv_features, final_target_features

def generate_multiple_hypos(dict_of_models, hype, device, decoder_input, decoder_hidden,
                            multi_agent_output, full_img_encoder_outputs, final_target_features, total_best_loss,
                            final_loss, time_step):
    decoder_outputs, decoder_hidden = dict_of_models['DECODER MODEL'](decoder_input.to(device),
                                                                      decoder_hidden.to(device),
                                                                      multi_agent_output.to(device),
                                                                      full_img_encoder_outputs.to(device))

    decoder_hidden_combined = torch.zeros(decoder_hidden[0, :, ].size()).to(device)
    for n in range(hype['decoder_n_layers']):
        decoder_hidden_combined += decoder_hidden[n, :, ]
    cvae_input = torch.cat((decoder_outputs, decoder_hidden_combined), 1)
    cvae_output, _, _ = dict_of_models['MULTI HYPO CVAE MODEL'](cvae_input.to(device))

    hypos = []
    for multi_hypo_model in dict_of_models['MULTI HYPO MODELS']:
        hypos.append(multi_hypo_model(cvae_output.to(device)))
    hypos.sort(key=lambda hypo: dict_of_models['LOSS FUNC'](final_target_features[time_step, :, ],
                                                            hypo + decoder_input.squeeze(0)).to(device))
    all_losses = []
    for hypo in hypos:
        temp_loss = dict_of_models['LOSS FUNC'](final_target_features[time_step, :, ], hypo + decoder_input.squeeze(0))
        if random.random() < hype['cvae_output_dropout']:
            temp_loss = 0
        all_losses.append(temp_loss)
    total_best_loss += all_losses[0]
    all_losses[0] = all_losses[0] * (1 - hype['epsilon'])
    all_losses[1:] = [(hype['epsilon'] / (hype['num_of_hypos'] - 1)) * the_loss for the_loss in all_losses]
    final_loss += sum(all_losses)

    return hypos, total_best_loss, final_loss, all_losses

def save_models(epoch, hype, dict_of_models, save_dir):
    torch.save({
        'epoch': epoch,
        'target_seq_len': hype['target_seq_len'],
        'input_seq_len': hype['input_seq_len'],
        'masked_output_ftrs': hype['masked_output_ftrs'],
        'mask_conv': dict_of_models['MASKED CONV MODEL'].state_dict(),
    }, os.path.join(save_dir, "MASK_CONV.tar"))
    torch.save({
        'unmasked_output_features': hype['unmasked_output_features'],
        'unmask_conv': dict_of_models['UNMASKED CONV MODEL'].state_dict(),
    }, os.path.join(save_dir, "UNMASK_CONV.tar"))
    model_counter = 0
    for multi_hypo_model in dict_of_models['MULTI HYPO MODELS']:
        torch.save({
            'multi_fc_layer_neurons': hype['multi_fc_layer_neurons'],
            'multi_hypo_' + str(model_counter): multi_hypo_model.state_dict(),
        }, os.path.join(save_dir, "MULTI_HYPO_" + str(model_counter) + ".tar"))
        model_counter += 1
    torch.save({
        'encoder_n_layers': hype['encoder_n_layers'],
        'encoder_hidden_size': hype['encoder_hidden_size'],
        'encoder_dropout': hype['encoder_dropout'],
        'encoder': dict_of_models['ENCODER MODEL'].state_dict(),
    }, os.path.join(save_dir, "ENCODER.tar"))
    torch.save({
        'decoder_n_layers': hype['decoder_n_layers'],
        'decoder_hidden_size': hype['decoder_hidden_size'],
        'decoder_dropout': hype['decoder_dropout'],
        'decoder': dict_of_models['DECODER MODEL'].state_dict(),
    }, os.path.join(save_dir, "DECODER.tar"))
    torch.save({
        'cvae_hidden_size': hype['cvae_hidden_size'],
        'cvae_latent_size': hype['cvae_latent_size'],
        'cvae': dict_of_models['CVAE MODEL'].state_dict(),
    }, os.path.join(save_dir, "CVAE.tar"))

def display_losses(loss_list, MASTER_ROOT_DIR):
    x = list(range(len(loss_list)))
    plt.xlabel("ITERATIONS")
    plt.ylabel("LOSSES")
    plt.title("LOSSES FOR EACH HYPOTHESIS AT THE OUTPUT OF EACH DECODER")
    for i in range(len(loss_list[0])):
        plt.plot(x, [loss[i] for loss in loss_list], label='hypothesis %s' % i)
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(MASTER_ROOT_DIR, 'losses.png'))