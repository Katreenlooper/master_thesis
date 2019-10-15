from custom_dataset import CustomDataSet
from network_init import createNetwork
from utils import *
from data_prep import *

def eval(MASTER_ROOT_DIR, script_args, hype, paths, transform, data, train_file,
        dict_of_models=None, device=None, DATASET=None, indices=None, epoch=None):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if DATASET is None:
        main_dir = os.path.join(MASTER_ROOT_DIR, script_args['ROOT_DIR'], script_args['TRAIN_DIR_MASKED'])
        DATASET = CustomDataSet(main_dir, transform=transform, json_data=data)
    if indices is None:
        indices = list(range(len(DATASET)))

    if dict_of_models is None:
        dict_of_models = createNetwork(hype, device)

    dict_of_models['MASKED CONV MODEL'].eval()
    dict_of_models['UNMASKED CONV MODEL'].eval()
    for multi_hypo_model in dict_of_models['MULTI HYPO MODELS']:
        multi_hypo_model.eval()
    dict_of_models['FULL IMG ENCODER MODEL'].eval()
    dict_of_models['MULTI AGENT ENCODER MODEL'].eval()
    dict_of_models['DECODER MODEL'].eval()

    unmasked_img_dir = os.path.join(MASTER_ROOT_DIR, script_args['ROOT_DIR'], script_args['TRAIN_DIR_UNMASKED'])
    encoder_batch_size = 1
    for index in range(script_args['DATA_AMOUNT_EVAL_START'], script_args['DATA_AMOUNT_EVAL_END']):
        batch_size = hype['input_seq_len']
        all_agents_unique, list_of_dicts, masked_img_array, masked_img_flipped_array, unmasked_img_array = \
            create_data(index, train_file, DATASET, indices, script_args, data, batch_size, MASTER_ROOT_DIR)
        for current_agent in all_agents_unique:
            current_agent_dict = getCurrentAgentBatches(current_agent, list_of_dicts, masked_img_array,
                                                        masked_img_flipped_array, unmasked_img_array, hype,
                                                        transform, device, unmasked_img_dir, (2, 2))

            if not current_agent_dict:
                print("EVAL: CURRENT AGENT " + str(current_agent) + " NOT FULLY PRESENT IN ANY BATCH!")
                train_file.write("EVAL: CURRENT AGENT " + str(current_agent) + " NOT FULLY PRESENT IN ANY BATCH!" + "\n")
            else:
                reference_agent_dict_list = []
                for reference_agent in all_agents_unique:
                    reference_agent_dict = getCurrentAgentBatches(current_agent, list_of_dicts, masked_img_array,
                                                                  masked_img_flipped_array, unmasked_img_array, hype,
                                                                  transform, device, unmasked_img_dir, (2, 2),
                                                                  reference_agent)
                    if reference_agent_dict:
                        print("REFERENCE AGENT " + str(reference_agent) + " PRESENT WITH AGENT " + str(current_agent))
                        train_file.write("REFERENCE AGENT " + str(reference_agent) + " PRESENT WITH AGENT " +
                                        str(current_agent) + "\n")
                        reference_agent_dict_list.append(reference_agent_dict)

                for batch, input in current_agent_dict.items():
                    # extract features and prepare input for encoder and decoder
                    mask_conv_output = dict_of_models['MASKED CONV MODEL'](input[0].to(device))
                    agent_conv_output = dict_of_models['UNMASKED CONV MODEL'](input[4].to(device))
                    agent_conv_output = torch.cat((agent_conv_output, input[1]), 1)
                    mask_conv_for_enc = convertToEncoderInput(mask_conv_output, hype['input_seq_len'],
                                                              hype['batch_size'])
                    agent_conv_for_enc = convertToEncoderInput(agent_conv_output, hype['input_seq_len'],
                                                               hype['batch_size'])
                    target_features_for_dec = convertToEncoderInput(input[3], hype['target_seq_len'],
                                                                    hype['batch_size'])

                    # encode full image sequence and agent image sequences
                    full_img_encoder_outputs, full_img_encoder_hidden = \
                        dict_of_models['FULL IMG ENCODER MODEL'](mask_conv_for_enc.to(device))
                    agent_encoder_outputs, agent_encoder_hidden = \
                        dict_of_models['MULTI AGENT ENCODER MODEL'](agent_conv_for_enc.to(device),
                                                                    full_img_encoder_outputs.to(device))
                    multi_agent_hidden = agent_encoder_hidden.clone()
                    multi_agent_output = agent_encoder_outputs.clone()
                    if reference_agent_dict_list:
                        for ref_agent_dict in reference_agent_dict_list:
                            if batch in ref_agent_dict.keys():
                                ref_agent_conv_output = \
                                    dict_of_models['UNMASKED CONV MODEL'](ref_agent_dict[batch][4].to(device))
                                ref_agent_conv_output = \
                                    torch.cat((ref_agent_conv_output, ref_agent_dict[batch][1]), 1)
                                ref_agent_conv_for_enc = convertToEncoderInput(ref_agent_conv_output,
                                                                               hype['input_seq_len'],
                                                                               hype['batch_size'])
                                enc_output, enc_hidden = \
                                    dict_of_models['MULTI AGENT ENCODER MODEL'](ref_agent_conv_for_enc.to(device),
                                                                                full_img_encoder_outputs.to(device))
                                multi_agent_hidden = multi_agent_hidden * enc_hidden
                                multi_agent_output = multi_agent_output * enc_output

                    encoder_hidden_final = multi_agent_hidden * full_img_encoder_hidden
                    decoder_input = input[2].unsqueeze(0)
                    decoder_hidden = encoder_hidden_final[:hype['decoder_n_layers']]

                    total_best_loss = 0
                    final_loss = 0

                    for t in range(hype['target_seq_len']):
                        hypos, total_best_loss, final_loss, all_losses = generate_multiple_hypos(dict_of_models, hype,
                                                                                                 device,
                                                                                                 decoder_input,
                                                                                                 decoder_hidden,
                                                                                                 multi_agent_output,
                                                                                                 target_features_for_dec,
                                                                                                 total_best_loss,
                                                                                                 final_loss, t)
                        decoder_input = hypos[0].unsqueeze(0) + decoder_input
                    if script_args['LOAD_MODEL']:
                        decoder_input = decoder_input.squeeze(0).cpu().numpy()
                    else:
                        decoder_input = decoder_input.squeeze(0).cpu().detach().numpy()

                    for dict in list_of_dicts[input[5]:input[6]]:
                        dict[str(current_agent) + ' (PREDICTION)'] = decoder_input

                    print("EVAL: THE ACCUMULATED BEST LOSS AT EPOCH " + str(epoch) + " IS: " + str(total_best_loss) +
                          " FOR BATCH " + str(batch) + " OF AGENT " + str(current_agent))
                    train_file.write(
                        "EVAL: THE ACCUMULATED BEST LOSS AT EPOCH " + str(epoch) + " IS: " + str(total_best_loss) +
                        " FOR BATCH " + str(batch) + " OF AGENT " + str(current_agent) + "\n")
            print("EVAL: COMPLETED AGENT " + str(current_agent))
            train_file.write("EVAL: COMPLETED AGENT " + str(current_agent) + "\n")

        with open(os.path.join(MASTER_ROOT_DIR, "predictions_network.txt"), "wb") as fp:
            pickle.dump(list_of_dicts, fp)