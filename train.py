import torch.nn as nn
from custom_dataset import CustomDataSet
from network_init import createNetwork
from utils import *
from data_prep import *
from eval import eval

def train(device, transform, flip_transform, data, MASTER_ROOT_DIR, trainFile, hype, script_args, paths):
    main_dir = os.path.join(MASTER_ROOT_DIR, script_args['ROOT_DIR'], script_args['TRAIN_DIR_MASKED'])
    DATASET = CustomDataSet(main_dir, transform=transform, json_data=data, flip_transform=flip_transform)
    indices = list(range(len(DATASET)))

    dict_of_models = createNetwork(hype, device)

    starting_epoch = 0
    losses_for_all_epochs = []
    unmasked_img_dir = os.path.join(MASTER_ROOT_DIR, script_args['ROOT_DIR'], script_args['TRAIN_DIR_UNMASKED'])
    save_dir = os.path.join(MASTER_ROOT_DIR, "saved models")
    batch_size = hype['batch_size'] + hype['input_seq_len'] - 1
    # if script_args['LOAD_MODEL']:
    #     starting_epoch = dict_of_models['EPOCH']
    #     hype['target_seq_len'] = dict_of_models['TARGET SEQ LEN']
    #     hype['input_seq_len'] = dict_of_models['INPUT SEQ LEN']
    for epoch in range(starting_epoch, hype['total_epochs']):
        for index in range(script_args['DATA_AMOUNT_TRAIN_START'], script_args['DATA_AMOUNT_TRAIN_END']):
            all_agents_unique, list_of_dicts, masked_img_array, masked_img_flipped_array, unmasked_img_array = \
                create_data(index, trainFile, DATASET, indices, script_args, data, batch_size, MASTER_ROOT_DIR)
            for current_agent in all_agents_unique:
                print("TRAIN: NOW WORKING ON AGENT " + str(current_agent) + " AT EPOCH " + str(epoch))
                trainFile.write("TRAIN: NOW WORKING ON AGENT " + str(current_agent) + " AT EPOCH " + str(epoch) + "\n")

                current_agent_dict = getCurrentAgentBatches(current_agent, list_of_dicts, masked_img_array,
                                                            masked_img_flipped_array, unmasked_img_array, hype,
                                                            transform, device, unmasked_img_dir,
                                                            (hype['horizontal_flip_chance'], hype['noise_chance']))

                if not current_agent_dict:
                    print("CURRENT AGENT " + str(current_agent) + " NOT FULLY PRESENT IN ANY BATCH!")
                    trainFile.write(
                        "CURRENT AGENT " + str(current_agent) + " NOT FULLY PRESENT IN ANY BATCH!" + "\n")
                else:
                    print("CURRENT AGENT " + str(current_agent) + " HAS BATCHES " + str(len(current_agent_dict.keys())))
                    trainFile.write("CURRENT AGENT " + str(current_agent) + " HAS BATCHES " +
                                    str(len(current_agent_dict.keys())) + "\n")

                    all_reference_agent_dicts = {}
                    for reference_agent in all_agents_unique:
                        if reference_agent != current_agent:
                            reference_agent_dict = getCurrentAgentBatches(current_agent, list_of_dicts, masked_img_array,
                                                                          masked_img_flipped_array, unmasked_img_array, hype,
                                                                          transform, device, unmasked_img_dir,
                                                                          (hype['horizontal_flip_chance'], hype['noise_chance']),
                                                                          reference_agent)
                            if reference_agent_dict:
                                print("REFERENCE AGENT " + str(reference_agent) + " PRESENT WITH AGENT " +
                                      str(current_agent))
                                trainFile.write("REFERENCE AGENT " + str(reference_agent) + " PRESENT WITH AGENT " +
                                                str(current_agent) + "\n")
                                all_reference_agent_dicts[reference_agent] = reference_agent_dict

                    for batch, input in sorted(current_agent_dict.items(), key=lambda x: random.random()):
                        #extract features and prepare input for encoder and decoder
                        mask_conv_output = dict_of_models['MASKED CONV MODEL'](input[0].to(device))
                        agent_conv_output = dict_of_models['UNMASKED CONV MODEL'](input[4].to(device))
                        agent_conv_output = torch.cat((agent_conv_output, input[1]), 1)
                        mask_conv_for_enc = convertToEncoderInput(mask_conv_output, hype['input_seq_len'],
                                                                  hype['batch_size'])
                        agent_conv_for_enc = convertToEncoderInput(agent_conv_output, hype['input_seq_len'],
                                                                   hype['batch_size'])
                        current_agent_target_features_for_dec = convertToEncoderInput(input[3], hype['target_seq_len'],
                                                                                      hype['batch_size'])

                        #encode full image sequence and agent image sequences
                        full_img_encoder_outputs, full_img_encoder_hidden = \
                            dict_of_models['FULL IMG ENCODER MODEL'](mask_conv_for_enc.to(device))
                        agent_encoder_outputs, agent_encoder_hidden = \
                            dict_of_models['MULTI AGENT ENCODER MODEL'](agent_conv_for_enc.to(device),
                                                                        full_img_encoder_outputs.to(device))
                        multi_agent_hidden = agent_encoder_hidden.clone()
                        multi_agent_output = agent_encoder_outputs.clone()
                        decoder_inputs_ground_truths = {}
                        decoder_inputs_ground_truths[current_agent] = \
                            (input[2].unsqueeze(0), current_agent_target_features_for_dec)
                        if all_reference_agent_dicts:
                            for ref_agent, ref_agent_dict in all_reference_agent_dicts.items():
                                if batch in ref_agent_dict.keys():
                                    print("REFERENCE AGENT " + str(ref_agent) + " PRESENT IN BATCH " +
                                          str(batch) + " WITH CURRENT AGENT " + str(current_agent))
                                    trainFile.write("REFERENCE AGENT " + str(ref_agent) + " PRESENT IN BATCH " +
                                          str(batch) + " WITH CURRENT AGENT " + str(current_agent) + "\n")

                                    ref_agent_conv_output = \
                                        dict_of_models['UNMASKED CONV MODEL'](ref_agent_dict[batch][4].to(device))
                                    ref_agent_conv_output = \
                                        torch.cat((ref_agent_conv_output, ref_agent_dict[batch][1]), 1)
                                    ref_agent_conv_for_enc = convertToEncoderInput(ref_agent_conv_output,
                                                                                   hype['input_seq_len'],
                                                                                   hype['batch_size'])
                                    ref_agent_target_features_for_dec = convertToEncoderInput(ref_agent_dict[batch][3],
                                                                                              hype['target_seq_len'],
                                                                                              hype['batch_size'])
                                    enc_output, enc_hidden = \
                                        dict_of_models['MULTI AGENT ENCODER MODEL'](ref_agent_conv_for_enc.to(device),
                                                                                    full_img_encoder_outputs.to(device))
                                    multi_agent_hidden = multi_agent_hidden * enc_hidden
                                    multi_agent_output = multi_agent_output * enc_output
                                    decoder_inputs_ground_truths[current_agent] = \
                                        (ref_agent_dict[batch][2].unsqueeze(0), ref_agent_target_features_for_dec)

                        dict_of_models['MASKED CONV MODEL OPTIMIZER'].zero_grad()
                        dict_of_models['UNMASKED CONV MODEL OPTIMIZER'].zero_grad()
                        for multi_hypo_optim in dict_of_models['MULTI HYPO MODEL OPTIMIZERS']:
                            multi_hypo_optim.zero_grad()
                        dict_of_models['FULL IMG ENCODER MODEL OPTIMIZER'].zero_grad()
                        dict_of_models['MULTI AGENT ENCODER MODEL OPTIMIZER'].zero_grad()
                        dict_of_models['DECODER MODEL OPTIMIZER'].zero_grad()
                        dict_of_models['MULTI HYPO CVAE MODEL OPTIMIZER'].zero_grad()

                        encoder_hidden_final = multi_agent_hidden * full_img_encoder_hidden
                        decoder_hidden = encoder_hidden_final[:hype['decoder_n_layers']]

                        #decode and predict future bbox for current agent and reference agents
                        for theAgent, (decoder_input, target_features_for_dec) in \
                                sorted(decoder_inputs_ground_truths.items(), key=lambda x: random.random()):
                            total_best_loss = 0
                            final_loss = 0

                            for t in range(hype['target_seq_len']):
                                hypos, total_best_loss, final_loss, all_losses = \
                                    generate_multiple_hypos(dict_of_models, hype, device, decoder_input, decoder_hidden,
                                                            multi_agent_output, full_img_encoder_outputs,
                                                            target_features_for_dec, total_best_loss, final_loss, t)
                                losses_for_all_epochs.append(all_losses)
                                if random.random() > hype['teacher_forcing_ratio']:
                                    decoder_input = target_features_for_dec[t, :, ].unsqueeze(0)
                                else:
                                    decoder_input = hypos[0].unsqueeze(0) + decoder_input

                                if random.random() > hype['noise_chance']:
                                    decoder_input = decoder_input + torch.randint_like(decoder_input, low=-10, high=10)

                            print("TRAIN: THE ACCUMULATED BEST LOSS AT EPOCH " + str(epoch) + " IS: " + str(total_best_loss) +
                                  " FOR BATCH " + str(batch) + " OF AGENT " + str(theAgent))
                            trainFile.write(
                                "TRAIN: THE ACCUMULATED BEST LOSS AT EPOCH " + str(epoch) + " IS: " + str(total_best_loss) +
                                " FOR BATCH " + str(batch) + " OF AGENT " + str(theAgent) + "\n")

                            # Perform backpropatation
                            final_loss.backward()

                            # Clip gradients: gradients are modified in place
                            _ = nn.utils.clip_grad_norm_(dict_of_models['FULL IMG ENCODER MODEL'].parameters(), hype['clip'])
                            _ = nn.utils.clip_grad_norm_(dict_of_models['MULTI AGENT ENCODER MODEL'].parameters(), hype['clip'])
                            _ = nn.utils.clip_grad_norm_(dict_of_models['DECODER MODEL'].parameters(), hype['clip'])

                            # Adjust model weights
                            dict_of_models['MASKED CONV MODEL OPTIMIZER'].step()
                            dict_of_models['UNMASKED CONV MODEL OPTIMIZER'].step()
                            for multi_hypo_optim in dict_of_models['MULTI HYPO MODEL OPTIMIZERS']:
                                multi_hypo_optim.step()
                            dict_of_models['FULL IMG ENCODER MODEL OPTIMIZER'].step()
                            dict_of_models['MULTI AGENT ENCODER MODEL OPTIMIZER'].step()
                            dict_of_models['DECODER MODEL OPTIMIZER'].step()
                            dict_of_models['MULTI HYPO CVAE MODEL OPTIMIZER'].step()

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