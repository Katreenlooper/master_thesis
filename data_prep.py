import random
import cv2
import torch
import os
from utils import flip_bbox
from PIL import Image

def getCurrentAgentBatches(current_agent, list_of_dicts, masked_img_array, masked_img_flipped_array, unmasked_img_array,
                           hype, transform, device, unmasked_img_dir, random_vals, reference_agent = None, eval=False):
    if reference_agent is not None:
        agent = reference_agent
    else:
        agent = current_agent

    horizontal_flip = False
    if random.random() > random_vals[0]:
        horizontal_flip = True

    size_of_input_batch = hype['batch_size'] + hype['input_seq_len'] - 1
    size_of_target_batch = hype['batch_size'] + hype['target_seq_len'] - 1

    batch_num = 0
    dict_of_batches = {}
    while True:
        agent_bbox_list = []
        agent_img_list = []
        target_bbox_list = []
        decoder_input_bbox_list = []

        starting_image = batch_num * size_of_input_batch
        ending_image = starting_image + size_of_input_batch
        input_img_starting_index = starting_image
        if (ending_image <= len(list_of_dicts)):
            skip = False
            image_counter = starting_image
            for dict in list_of_dicts[starting_image:ending_image]:
                #get bounding box and image of agent from input sequence
                full_img = cv2.imread(os.path.join(unmasked_img_dir, unmasked_img_array[image_counter]))
                if horizontal_flip:
                    full_img = cv2.flip(full_img, 1)

                if check_presence(current_agent, reference_agent, dict):
                    if horizontal_flip:
                        bbox_tensor = torch.from_numpy(flip_bbox(hype['horizontal_img_size']/2, dict[agent]))
                        agent_bbox = [abs(i) for i in flip_bbox(hype['horizontal_img_size']/2, dict[agent])]
                    else:
                        bbox_tensor = torch.from_numpy(dict[agent])
                        agent_bbox = [abs(i) for i in dict[agent]]

                    if random.random() > random_vals[1] and \
                            10 < agent_bbox[0] < full_img.shape[0] - 10 and \
                            10 < agent_bbox[1] < full_img.shape[1] - 10 and \
                            10 < agent_bbox[2] < full_img.shape[0] - 10 and \
                            10 < agent_bbox[3] < full_img.shape[1] - 10:
                        bbox_tensor = bbox_tensor + torch.randint_like(bbox_tensor, low=-10, high=10)
                        rand_value = random.randrange(-10, 10)
                        agent_bbox = [x + rand_value for x in agent_bbox]

                    agent_bbox_list.append(bbox_tensor.float().unsqueeze(0))
                    agent_img_list.append(transform(Image.fromarray(full_img
                                                                    [int(agent_bbox[1]):int(agent_bbox[3]),
                                                                    int(agent_bbox[0]):int(agent_bbox[2])])).unsqueeze(0))
                image_counter += 1

            if len(agent_bbox_list) != size_of_input_batch:
                skip = True

            if not skip:
                #get the bbox and image of agent from every last image in each of your image sequences
                starting_image += hype['input_seq_len'] - 1
                for dict in list_of_dicts[starting_image:ending_image]:
                    if horizontal_flip:
                        bbox_tensor = torch.from_numpy(flip_bbox(hype['horizontal_img_size']/2, dict[agent]))
                    else:
                        bbox_tensor = torch.from_numpy(dict[agent])

                    if random.random() > random_vals[1]:
                        bbox_tensor = bbox_tensor + torch.randint_like(bbox_tensor, low=-10, high=10)

                    decoder_input_bbox_list.append(bbox_tensor.float().unsqueeze(0))

                #get the bbox for target images
                starting_image += 1
                ending_image = starting_image + size_of_target_batch
                if ending_image < len(list_of_dicts):
                    for dict in list_of_dicts[starting_image:ending_image]:
                        if check_presence(current_agent, reference_agent, dict):
                            if horizontal_flip:
                                bbox_tensor = torch.from_numpy(flip_bbox(hype['horizontal_img_size']/2, dict[agent]))
                            else:
                                bbox_tensor = torch.from_numpy(dict[agent])

                            if random.random() > random_vals[1]:
                                bbox_tensor = bbox_tensor + torch.randint_like(bbox_tensor, low=-10, high=10)

                            target_bbox_list.append(bbox_tensor.float().unsqueeze(0))

            if len(target_bbox_list) != size_of_target_batch:
                skip = True

            #create dictionary for current agent indexed by batch number
            if not skip:
                agent_bbox_tensor = torch.cat(agent_bbox_list).to(device)
                agent_img_tensor = torch.cat(agent_img_list).to(device)
                input_bbox_tensor = torch.cat(decoder_input_bbox_list).to(device)
                target_bbox_tensor = torch.cat(target_bbox_list).to(device)
                masked_img_batch = masked_img_flipped_array[batch_num] if horizontal_flip else masked_img_array[batch_num]
                if not eval:
                    dict_of_batches[batch_num] = (masked_img_batch, agent_bbox_tensor, input_bbox_tensor,
                                                  target_bbox_tensor, agent_img_tensor)
                else:
                    dict_of_batches[batch_num] = (masked_img_batch, agent_bbox_tensor, input_bbox_tensor,
                                                  target_bbox_tensor, agent_img_tensor, input_img_starting_index, ending_image)
            batch_num += 1
        else:
            break

    return dict_of_batches

def check_presence(current_agent, reference_agent, dict):
    if current_agent in dict and reference_agent is None:
        return True
    elif reference_agent is not None and (current_agent and reference_agent in dict):
        return True