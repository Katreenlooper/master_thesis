import torchvision.models as models
from torch import optim
from encoder_decoder_rnn import *
from CVAE import CVAE

def create_custom_model(output_ftrs, lr):
    params = []
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, output_ftrs), nn.ReLU(), nn.Dropout())
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params.append(param)
    optimizer = optim.Adam(params, lr)

    return model, params, optimizer

def create_new_model(input_size, output_size, fc_layer_neurons, lr, device):
    fc_model = nn.Sequential(
        nn.Linear(input_size, fc_layer_neurons[0]), nn.ReLU(), nn.Dropout(),
        nn.Linear(fc_layer_neurons[0], fc_layer_neurons[1]), nn.ReLU(), nn.Dropout(),
        nn.Linear(fc_layer_neurons[1], output_size), nn.ReLU(), nn.Dropout()
    )
    fc_params = []
    for name, param in fc_model.named_parameters():
        fc_params.append(param)
    optimizer = optim.Adam(fc_params, lr=lr)
    fc_model = fc_model.to(device)

    return fc_model, fc_params, optimizer

#use following formula (https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw/136542#136542)
#num of hidden neurons = (number of samples in training data) / (alpha * (number of input neurons + number of output neurons))

def createNetwork(hype, device):
    multi_hypo_models = []
    multi_hypo_optimizers = []
    multi_hypo_params = []

    mask_conv_model, mask_params, mask_conv_model_optimizer = create_custom_model(hype['masked_output_ftrs'],
                                                                                  hype['masked_learning_rate'])
    other_agent_model, other_agent_params, other_agent_model_optimizer = \
        create_custom_model(hype['unmasked_output_ftrs'], hype['unmasked_learning_rate'])
    mask_conv_model = mask_conv_model.to(device)
    other_agent_model = other_agent_model.to(device)

    for hypo in range(hype['num_of_hypos']):
        fc_model, fc_params, multi_hypo_optimizer = \
            create_new_model(input_size=4 + 125, output_size=4, fc_layer_neurons=hype['multi_fc_layer_neurons'],
                             lr=hype['multi_hypo_learning_rate'], device=device)
        multi_hypo_models.append(fc_model)
        multi_hypo_params.append(fc_params)
        multi_hypo_optimizers.append(multi_hypo_optimizer)

    full_img_encoder = EncoderRNN(input_size=hype['masked_output_ftrs'], hidden_size=125, n_layers=1, dropout=0.1)
    multi_agent_encoder = LuongAttnEncoderRNN(hype['attn_model'], input_size=hype['unmasked_output_ftrs'] + 4,
                                              hidden_size=125, n_layers=1, dropout=0.1)
    decoder = LuongAttnDecoderRNN(hype['attn_model'], input_size=4, hidden_size=125,
                                  output_size=4, n_layers=1, dropout=0.1)
    multi_hypo_cvae = CVAE(input_dim=4 + 125, output_dim=4 + 125, hidden_dim=20, latent_dim=20)

    full_img_encoder = full_img_encoder.to(device)
    multi_agent_encoder = multi_agent_encoder.to(device)
    decoder = decoder.to(device)
    multi_hypo_cvae = multi_hypo_cvae.to(device)

    full_img_encoder_optimizer = optim.Adam(full_img_encoder.parameters(), lr=hype['encoder_learning_rate'])
    multi_agent_encoder_optimizer = optim.Adam(multi_agent_encoder.parameters(), lr=hype['encoder_learning_rate'])
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=hype['decoder_learning_rate'])
    multi_hypo_cvae_optimizer = optim.Adam(multi_hypo_cvae.parameters(), lr=hype['cvae_learning_rate'])

    loss = nn.MSELoss()

    ret_vals = {}
    ret_vals['MASKED CONV MODEL'] = mask_conv_model
    ret_vals['MASKED CONV MODEL OPTIMIZER'] = mask_conv_model_optimizer
    ret_vals['UNMASKED CONV MODEL'] = other_agent_model
    ret_vals['UNMASKED CONV MODEL OPTIMIZER'] = other_agent_model_optimizer
    ret_vals['MULTI HYPO MODELS'] = multi_hypo_models
    ret_vals['MULTI HYPO MODEL OPTIMIZERS'] = multi_hypo_optimizers
    ret_vals['FULL IMG ENCODER MODEL'] = full_img_encoder
    ret_vals['MULTI AGENT ENCODER MODEL'] = multi_agent_encoder
    ret_vals['FULL IMG ENCODER MODEL OPTIMIZER'] = full_img_encoder_optimizer
    ret_vals['MULTI AGENT ENCODER MODEL OPTIMIZER'] = multi_agent_encoder_optimizer
    ret_vals['DECODER MODEL'] = decoder
    ret_vals['DECODER MODEL OPTIMIZER'] = decoder_optimizer
    ret_vals['MULTI HYPO CVAE MODEL'] = multi_hypo_cvae
    ret_vals['MULTI HYPO CVAE MODEL OPTIMIZER'] = multi_hypo_cvae_optimizer
    ret_vals['LOSS FUNC'] = loss
    ret_vals['EPOCH'] = 0
    ret_vals['TARGET SEQ LEN'] = hype['target_seq_len']
    ret_vals['INPUT SEQ LEN'] = hype['input_seq_len']

    return ret_vals
    # else:
    #     mask_checkpoint = torch.load(os.path.join(save_dir, paths['MASK_CONV']))
    #     epoch = mask_checkpoint['epoch']
    #     masked_output_ftrs = mask_checkpoint['masked_output_ftrs']
    #     num_ftrs = mask_conv_model.fc.in_features
    #     mask_conv_model.fc = nn.Linear(num_ftrs, masked_output_ftrs)
    #     mask_conv_model.load_state_dict(mask_checkpoint['mask_conv'])
    #     input_seq_len = mask_checkpoint['input_seq_len']
    #     target_seq_len = mask_checkpoint['target_seq_len']
    #     print("RECENT EPOCH WAS " + str(epoch))
    #     if script_args['ACTION'] == "TRAIN":
    #         for name, param in mask_conv_model.named_parameters():
    #             if param.requires_grad == True:
    #                 mask_params.append(param)
    #         mask_conv_model_optimizer = optim.Adam(mask_params, lr=hype['masked_learning_rate'])
    #     else:
    #         for param in mask_conv_model.parameters():
    #             param.requires_grad = False
    #
    #     unmask_checkpoint = torch.load(os.path.join(save_dir, paths['UNMASK_CONV']))
    #     unmasked_output_ftrs = unmask_checkpoint['unmasked_output_ftrs']
    #     num_ftrs = other_agent_model.fc.in_features
    #     other_agent_model.fc = nn.Linear(num_ftrs, unmasked_output_ftrs)
    #     other_agent_model.load_state_dict(unmask_checkpoint['unmask_conv'])
    #     if script_args['ACTION'] == "TRAIN":
    #         for name, param in other_agent_model.named_parameters():
    #             if param.requires_grad == True:
    #                 other_agent_params.append(param)
    #         other_agent_model_optimizer = optim.Adam(other_agent_params, lr=hype['unmasked_learning_rate'])
    #     else:
    #         for param in other_agent_model.parameters():
    #             param.requires_grad = False
    #
    #     dec = torch.load(os.path.join(save_dir, paths['DECODER']))
    #     decoder_hidden_size = dec['decoder_hidden_size']
    #     for hypo in range(hype['num_of_hypos']):
    #         multi_hypo_checkpoint = torch.load(os.path.join(save_dir, multi_hypo_dir_list[hypo]))
    #         fc_model = nn.Sequential(
    #             nn.Linear(4 + decoder_hidden_size, multi_fc_layer_neurons[0]), nn.ReLU(), nn.Dropout(),
    #             nn.Linear(multi_fc_layer_neurons[0], multi_fc_layer_neurons[1]), nn.ReLU(), nn.Dropout(),
    #             nn.Linear(multi_fc_layer_neurons[1], multi_fc_layer_neurons[2]), nn.ReLU(), nn.Dropout(),
    #             nn.Linear(multi_fc_layer_neurons[2], multi_fc_layer_neurons[3]), nn.ReLU(), nn.Dropout(),
    #             nn.Linear(multi_fc_layer_neurons[3], multi_fc_layer_neurons[4]), nn.ReLU(), nn.Dropout(),
    #             nn.Linear(multi_fc_layer_neurons[4], 4), nn.ReLU(), nn.Dropout()
    #         )
    #         fc_model.load_state_dict(multi_hypo_checkpoint['multi_hypo_' + str(hypo)]).to(device)
    #         multi_hypo_models.append(fc_model)
    #         if train:
    #             fc_params = []
    #             for name, param in fc_model.named_parameters():
    #                 fc_params.append(param)
    #             multi_hypo_optimizer = optim.Adam(fc_params, lr=multi_hypo_learning_rate)
    #             multi_hypo_params.append(fc_params)
    #             multi_hypo_optimizers.append(multi_hypo_optimizer)
    #         else:
    #             for param in fc_model.parameters():
    #                 param.requires_grad = False
    #
    # if load_model:
    #     enc_checkpoint = torch.load(os.path.join(save_dir, enc_dir))
    #     encoder_n_layers = enc_checkpoint['encoder_n_layers']
    #     encoder_hidden_size = enc_checkpoint['encoder_hidden_size']
    #     encoder_dropout = enc_checkpoint['encoder_dropout']
    #     encoder = EncoderRNN(input_size=masked_output_ftrs + unmasked_output_ftrs + 4,
    #                          hidden_size=encoder_hidden_size, n_layers=encoder_n_layers, dropout=encoder_dropout)
    #     encoder.load_state_dict(enc_checkpoint['encoder'])
    #
    #     dec_checkpoint = torch.load(os.path.join(save_dir, dec_dir))
    #     decoder_n_layers = dec_checkpoint['decoder_n_layers']
    #     decoder_hidden_size = dec_checkpoint['decoder_hidden_size']
    #     decoder_dropout = dec_checkpoint['decoder_dropout']
    #     decoder = LuongAttnDecoderRNN(attn_model, input_size=4, hidden_size=decoder_hidden_size, output_size=4,
    #                                   n_layers=decoder_n_layers, dropout=decoder_dropout)
    #     decoder.load_state_dict(dec_checkpoint['decoder'])
    #
    #     cvae_checkpoint = torch.load(cvae_dir)
    #     cvae_hidden_size = cvae_checkpoint['cvae_hidden_size']
    #     cvae_latent_size = cvae_checkpoint['cvae_latent_size']
    #     cvae = CVAE(input_dim=4 + decoder_hidden_size, hidden_dim=cvae_hidden_size, latent_dim=cvae_latent_size)
    #     cvae.load_state_dict(cvae_checkpoint['cvae'])
    #
    # encoder_optimizer = []
    # decoder_optimizer = []
    # cvae_optimizer = []
    #
    # if train:
    #
    # else:
    #     for param in encoder.parameters():
    #         param.requires_grad = False
    #     for param in decoder.parameters():
    #         param.requires_grad = False
    #     for param in cvae.parameters():
    #         param.requires_grad = False