from custom_dataset import CustomDataSet
from network_init import createNetwork
from utils import *
from data_prep import create_data_dict

def eval(MASTER_ROOT_DIR, script_args, hype, paths, transform, data, train_file,
        dict_of_models=None, device=None, DATASET=None, indices=None, epoch=None):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if DATASET is None:
        DATASET = CustomDataSet(os.path.join(MASTER_ROOT_DIR, script_args['ROOT_DIR'], script_args['TRAIN_DIR_MASKED']),
                                transform=transform, json_data=data)
    if indices is None:
        indices = list(range(len(DATASET)))

    if dict_of_models is None:
        dict_of_params = create_model_dict(hype, device)
        dict_of_models = createNetwork(dict_of_params, paths, hype['input_seq_len'], hype['target_seq_len'],
                                       script_args['LOAD_MODEL'], script_args['TRAIN'])

    dict_of_models['MASKED CONV MODEL'].eval()
    dict_of_models['UNMASKED CONV MODEL'].eval()
    for multi_hypo_model in dict_of_models['MULTI HYPO MODELS']:
        multi_hypo_model.eval()
    dict_of_models['ENCODER MODEL'].eval()
    dict_of_models['DECODER MODEL'].eval()

    for index in range(script_args['DATA_AMOUNT_EVAL_START'], script_args['DATA_AMOUNT_EVAL_END']):
        batch_size = hype['input_seq_len']
        all_agents_unique, list_of_dicts, masked_img_array, masked_img_flipped_array, unmasked_img_array = \
            create_data(index, train_file, DATASET, indices, script_args, data, batch_size, MASTER_ROOT_DIR)
        for current_agent in all_agents_unique:
            encoder_batch_size = 1
            noise_chance = 2
            horizontal_flip_chance = 2
            log_type = "EVAL"
            current_agent_dict = create_data_dict(current_agent, epoch, train_file, script_args, list_of_dicts,
                                                  masked_img_array, masked_img_flipped_array, unmasked_img_array,
                                                  hype, transform, device, MASTER_ROOT_DIR, encoder_batch_size,
                                                  noise_chance, horizontal_flip_chance, log_type, eval=True)

            if not current_agent_dict:
                print("EVAL: CURRENT AGENT " + str(current_agent) + " NOT FULLY PRESENT IN ANY BATCH!")
                train_file.write("EVAL: CURRENT AGENT " + str(current_agent) + " NOT FULLY PRESENT IN ANY BATCH!" + "\n")
            else:
                for batch, input in current_agent_dict.items():
                    final_mask_conv_features, final_target_features = \
                        feature_extraction(dict_of_models, device, hype, encoder_batch_size, input)

                    encoder_outputs, encoder_hidden_final = dict_of_models['ENCODER MODEL'](final_mask_conv_features.to(device))

                    hidden_to_decoder = encoder_hidden_final[:hype['decoder_n_layers']]

                    decoder_input = input[2].unsqueeze(0)
                    decoder_hidden = hidden_to_decoder

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