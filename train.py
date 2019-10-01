import torch.nn as nn
from custom_dataset import CustomDataSet
from network_init import createNetwork
from utils import *
from data_prep import create_data_dict
from eval import eval

def train(device, transform, flip_transform, data, MASTER_ROOT_DIR, trainFile, hype, script_args, paths):
    DATASET = CustomDataSet(os.path.join(MASTER_ROOT_DIR, script_args['ROOT_DIR'], script_args['TRAIN_DIR_MASKED']),
                            transform=transform, json_data=data, flip_transform=flip_transform)
    indices = list(range(len(DATASET)))

    dict_of_params = create_model_dict(hype, device)
    save_dir = os.path.join(MASTER_ROOT_DIR, "saved models")
    dict_of_models = createNetwork(dict_of_params, paths, hype['input_seq_len'], hype['target_seq_len'], save_dir,
                                   script_args['LOAD_MODEL'], script_args['TRAIN'])

    starting_epoch = 0
    losses_for_all_epochs = []
    if script_args['LOAD_MODEL']:
        starting_epoch = dict_of_models['EPOCH']
        hype['target_seq_len'] = dict_of_models['TARGET SEQ LEN']
        hype['input_seq_len'] = dict_of_models['INPUT SEQ LEN']
    for epoch in range(starting_epoch, hype['total_epochs']):
        for index in range(script_args['DATA_AMOUNT_TRAIN_START'], script_args['DATA_AMOUNT_TRAIN_END']):
            batch_size = hype['batch_size'] + hype['input_seq_len'] - 1
            all_agents_unique, list_of_dicts, masked_img_array, masked_img_flipped_array, unmasked_img_array = \
                create_data(index, trainFile, DATASET, indices, script_args, data, batch_size, MASTER_ROOT_DIR)
            for current_agent in all_agents_unique:
                encoder_batch_size = hype['batch_size']
                noise_chance = hype['noise_chance']
                horizontal_flip_chance = hype['horizontal_flip_chance']
                log_type = "TRAIN"
                current_agent_dict = create_data_dict(current_agent, epoch, trainFile, script_args, list_of_dicts,
                                                     masked_img_array, masked_img_flipped_array, unmasked_img_array,
                                                     hype, transform, device, MASTER_ROOT_DIR, encoder_batch_size,
                                                     noise_chance, horizontal_flip_chance, log_type)

                if not current_agent_dict:
                    print("CURRENT AGENT " + str(current_agent) + " NOT FULLY PRESENT IN ANY BATCH!")
                    trainFile.write(
                        "CURRENT AGENT " + str(current_agent) + " NOT FULLY PRESENT IN ANY BATCH!" + "\n")
                else:
                    for batch, input in sorted(current_agent_dict.items(), key=lambda x: random.random()):
                        final_mask_conv_features, final_target_features = \
                            feature_extraction(dict_of_models, device, hype, encoder_batch_size, input)
                        encoder_outputs, encoder_hidden_final = \
                            dict_of_models['ENCODER MODEL'](final_mask_conv_features.to(device))

                        dict_of_models['MASKED CONV MODEL OPTIMIZER'].zero_grad()
                        dict_of_models['UNMASKED CONV MODEL OPTIMIZER'].zero_grad()
                        for multi_hypo_optim in dict_of_models['MULTI HYPO MODEL OPTIMIZERS']:
                            multi_hypo_optim.zero_grad()
                        dict_of_models['ENCODER MODEL OPTIMIZER'].zero_grad()
                        dict_of_models['DECODER MODEL OPTIMIZER'].zero_grad()
                        dict_of_models['CVAE MODEL OPTIMIZER'].zero_grad()

                        decoder_input = input[2].unsqueeze(0)
                        decoder_hidden = encoder_hidden_final[:hype['decoder_n_layers']]

                        total_best_loss = 0
                        final_loss = 0

                        for t in range(hype['target_seq_len']):
                            hypos, total_best_loss, final_loss, all_losses = generate_multiple_hypos(dict_of_models, hype,
                                                                                                     device,
                                                                                                     decoder_input,
                                                                                                     decoder_hidden,
                                                                                                     encoder_outputs,
                                                                                                     final_target_features,
                                                                                                     total_best_loss,
                                                                                                     final_loss, t)
                            losses_for_all_epochs.append(all_losses)
                            if random.random() > hype['teacher_forcing_ratio']:
                                decoder_input = final_target_features[t, :, ].unsqueeze(0)
                            else:
                                decoder_input = hypos[0].unsqueeze(0) + decoder_input

                            if random.random() > hype['noise_chance']:
                                decoder_input = decoder_input + torch.randint_like(decoder_input, low=-10, high=10)

                        print("TRAIN: THE ACCUMULATED BEST LOSS AT EPOCH " + str(epoch) + " IS: " + str(total_best_loss) +
                              " FOR BATCH " + str(batch) + " OF AGENT " + str(current_agent))
                        trainFile.write(
                            "TRAIN: THE ACCUMULATED BEST LOSS AT EPOCH " + str(epoch) + " IS: " + str(total_best_loss) +
                            " FOR BATCH " + str(batch) + " OF AGENT " + str(current_agent) + "\n")

                        # Perform backpropatation
                        final_loss.backward()

                        # Clip gradients: gradients are modified in place
                        _ = nn.utils.clip_grad_norm_(dict_of_models['ENCODER MODEL'].parameters(), hype['clip'])
                        _ = nn.utils.clip_grad_norm_(dict_of_models['DECODER MODEL'].parameters(), hype['clip'])

                        # Adjust model weights
                        dict_of_models['MASKED CONV MODEL OPTIMIZER'].step()
                        dict_of_models['UNMASKED CONV MODEL OPTIMIZER'].step()
                        for multi_hypo_optim in dict_of_models['MULTI HYPO MODEL OPTIMIZERS']:
                            multi_hypo_optim.step()
                        dict_of_models['ENCODER MODEL OPTIMIZER'].step()
                        dict_of_models['DECODER MODEL OPTIMIZER'].step()
                        dict_of_models['CVAE MODEL OPTIMIZER'].step()

                print("TRAIN: COMPLETED AGENT " + str(current_agent) + " AT EPOCH " + str(epoch))
                trainFile.write("TRAIN: COMPLETED AGENT " + str(current_agent) + " AT EPOCH " + str(epoch) + "\n")
                trainFile.write("########################################################################" + "\n")
        print("TRAIN: COMPLETED EPOCH " + str(epoch))
        trainFile.write("TRAIN: COMPLETED EPOCH " + str(epoch) + "\n")
        print("TRAIN: NOW SAVING MODELS AND LOSSES...")
        trainFile.write("TRAIN: NOW SAVING MODELS AND LOSSES..." + "\n")
        trainFile.write("============================================================" + "\n")
        trainFile.write("============================================================" + "\n")
        trainFile.write("============================================================" + "\n")
        save_models(epoch, hype, dict_of_models, save_dir)
        with open(os.path.join(MASTER_ROOT_DIR, "losses.txt"), "wb") as fp:
            pickle.dump(losses_for_all_epochs, fp)
        print("NOW ATTEMPTING EVAL")
        trainFile.write("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||" + "\n")
        trainFile.write("NOW ATTEMPTING EVAL" + "\n")
        eval(MASTER_ROOT_DIR, script_args, hype, paths, transform, data, trainFile,
             dict_of_models, device, DATASET, indices, epoch)
        print("EVAL COMPLETED")
        trainFile.write("EVAL COMPLETED" + "\n")
        trainFile.write("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||" + "\n")
    trainFile.close()