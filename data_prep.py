import random
import cv2
import torch
import os
from utils import flip_bbox
from PIL import Image

def getCurrentAgentBatches(dict_of_params, eval=False):
    current_agent = dict_of_params['CURRENT AGENT']
    list_of_dicts = dict_of_params['LIST OF FRAME DICTS']
    (masked_img_array, masked_img_flipped_array) = dict_of_params['MASK IMG ARRAYS']
    (unmasked_img_array, unmasked_img_dir) = dict_of_params['UNMASK IMG ARRAY']
    (batch_size, input_seq_len, target_seq_len, transform, device,
     noise_chance, horizontal_flip_chance, horizontal_img_size) = dict_of_params['CURRENT AGENT MISC']

    horizontal_flip = False
    if random.random() > horizontal_flip_chance:
        horizontal_flip = True

    size_of_input_batch = batch_size + input_seq_len - 1
    size_of_target_batch = batch_size + target_seq_len - 1

    list_offset = 0
    dict_of_batches = {}
    while True:
        agent_bbox_encoder = []
        current_agent_img_encoder = []
        target_bbox = []
        input_bbox = []

        starting_image = list_offset * size_of_input_batch
        ending_image = starting_image + size_of_input_batch
        input_img_starting_index = starting_image
        if (ending_image <= len(list_of_dicts)):
            skip = False
            image_counter = starting_image
            for dict in list_of_dicts[starting_image:ending_image]:
                #get bounding box of agent whose trajectory you wish to predict
                current_agent_img = cv2.imread(os.path.join(unmasked_img_dir, unmasked_img_array[image_counter]))
                if horizontal_flip:
                    current_agent_img = cv2.flip(current_agent_img, 1)

                if current_agent in dict:
                    bbox_tensor = 0
                    if horizontal_flip:
                        bbox_tensor = torch.from_numpy(flip_bbox(horizontal_img_size, dict[current_agent]))
                    else:
                        bbox_tensor = torch.from_numpy(dict[current_agent])

                    if random.random() > noise_chance:
                        bbox_tensor = bbox_tensor + torch.randint_like(bbox_tensor, low=-10, high=10)

                    agent_bbox_encoder.append(bbox_tensor.float().unsqueeze(0))
                    current_agent_bbox = [abs(i) for i in dict[current_agent]]

                    current_agent_img_encoder.append(
                        transform(Image.fromarray(current_agent_img
                                                  [int(current_agent_bbox[1]):int(current_agent_bbox[3]),
                                                  int(current_agent_bbox[0]):int(current_agent_bbox[2])])).unsqueeze(0))
                image_counter += 1

            if len(agent_bbox_encoder) != size_of_input_batch:
                skip = True

            if not skip:
                #get the bbox for every last image in each of your image sequences
                starting_image += input_seq_len - 1
                for dict in list_of_dicts[starting_image:ending_image]:
                    bbox_tensor = 0
                    if horizontal_flip:
                        bbox_tensor = torch.from_numpy(flip_bbox(horizontal_img_size, dict[current_agent]))
                    else:
                        bbox_tensor = torch.from_numpy(dict[current_agent])

                    if random.random() > noise_chance:
                        bbox_tensor = bbox_tensor + torch.randint_like(bbox_tensor, low=-10, high=10)

                    input_bbox.append(bbox_tensor.float().unsqueeze(0))

                #get the bbox for target images
                starting_image += 1
                ending_image = starting_image + size_of_target_batch
                if ending_image < len(list_of_dicts):
                    for dict in list_of_dicts[starting_image:ending_image]:
                        if current_agent in dict:
                            bbox_tensor = 0
                            if horizontal_flip:
                                bbox_tensor = torch.from_numpy(flip_bbox(horizontal_img_size, dict[current_agent]))
                            else:
                                bbox_tensor = torch.from_numpy(dict[current_agent])

                            if random.random() > noise_chance:
                                bbox_tensor = bbox_tensor + torch.randint_like(bbox_tensor, low=-10, high=10)

                            target_bbox.append(bbox_tensor.float().unsqueeze(0))

            if len(target_bbox) != size_of_target_batch:
                skip = True

            #create dictionary for current agent indexed by batch number
            if not skip:
                agent_bbox_encoder_tensor = torch.cat(agent_bbox_encoder).to(device)
                current_agent_img_encoder_tensor = torch.cat(current_agent_img_encoder).to(device)
                input_bbox_tensor = torch.cat(input_bbox).to(device)
                target_bbox_tensor = torch.cat(target_bbox).to(device)
                masked_img_batch = masked_img_flipped_array[list_offset] if horizontal_flip else masked_img_array[list_offset]
                if not eval:
                    dict_of_batches[list_offset] = (masked_img_batch, agent_bbox_encoder_tensor,
                                                    input_bbox_tensor, target_bbox_tensor, current_agent_img_encoder_tensor)
                else:
                    dict_of_batches[list_offset] = (masked_img_batch, agent_bbox_encoder_tensor,
                                                    input_bbox_tensor, target_bbox_tensor,
                                                    current_agent_img_encoder_tensor,
                                                    input_img_starting_index, ending_image)
            list_offset += 1
        else:
            break

    return dict_of_batches

def create_data_dict(current_agent, epoch, trainFile, script_args, list_of_dicts,
                     masked_img_array, masked_img_flipped_array, unmasked_img_array,
                     hype, transform, device, MASTER_ROOT_DIR, encoder_batch_size,
                     noise_chance, horizontal_flip_chance, log_type, eval=False):
    print(log_type + ": " + "NOW WORKING ON AGENT " + str(current_agent) + " AT EPOCH " + str(epoch))
    trainFile.write(log_type + ": " + "NOW WORKING ON AGENT " + str(current_agent) + " AT EPOCH " + str(epoch) + "\n")
    unmasked_img_dir = os.path.join(MASTER_ROOT_DIR, script_args['ROOT_DIR'], script_args['TRAIN_DIR_UNMASKED'])

    dict_of_params = {}
    dict_of_params['CURRENT AGENT'] = current_agent
    dict_of_params['LIST OF FRAME DICTS'] = list_of_dicts
    dict_of_params['MASK IMG ARRAYS'] = (masked_img_array, masked_img_flipped_array)
    dict_of_params['UNMASK IMG ARRAY'] = (unmasked_img_array, unmasked_img_dir)
    dict_of_params['CURRENT AGENT MISC'] = (encoder_batch_size, hype['input_seq_len'], hype['target_seq_len'],
                                            transform, device, noise_chance, horizontal_flip_chance,
                                            hype['horizontal_img_size'])
    current_agent_dict = getCurrentAgentBatches(dict_of_params, eval)
    return current_agent_dict