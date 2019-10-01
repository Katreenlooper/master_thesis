import os
import torchvision.models as models
import torch
import torch.nn as nn
from torch import optim
from encoder_decoder_rnn import EncoderRNN, LuongAttnDecoderRNN
from CVAE import CVAE

def createNetwork(dict_of_params, paths, input_seq_len, target_seq_len, save_dir, load_model=False, train=True):
    (masked_output_ftrs, masked_learning_rate) = dict_of_params['MASKED CONV MODEL']
    (unmasked_output_ftrs, unmasked_learning_rate) = dict_of_params['UNMASKED CONV MODEL']
    (encoder_hidden_size, encoder_n_layers, encoder_dropout, encoder_learning_rate) = dict_of_params['ENCODER MODEL']
    (decoder_hidden_size, decoder_n_layers, decoder_dropout, decoder_learning_rate) = dict_of_params['DECODER MODEL']
    (cvae_hidden_size, cvae_latent_size, cvae_learning_rate) = dict_of_params['CVAE MODEL']
    (device, attn_model) = dict_of_params['MISC']
    (num_of_hypos, multi_fc_layer_neurons, multi_hypo_learning_rate) = dict_of_params['MULTI HYPOS']
    mask_dir = paths['MASK_CONV']
    unmask_dir = paths['UNMASK_CONV']
    enc_dir = paths['ENCODER']
    dec_dir = paths['DECODER']
    cvae_dir = paths['CVAE']
    multi_hypo_dir_list = paths['MULTI_HYPO']

    mask_conv_model = models.resnet50(pretrained=True)
    other_agent_model = models.resnet50(pretrained=True)
    multi_hypo_models = []
    mask_conv_model_optimizer = []
    other_agent_model_optimizer = []
    multi_hypo_optimizers = []
    mask_params = []
    other_agent_params = []
    multi_hypo_params = []
    epoch = 0
    if not load_model:
        for param in mask_conv_model.parameters():
            param.requires_grad = False
        num_ftrs = mask_conv_model.fc.in_features
        mask_conv_model.fc = nn.Sequential(nn.Linear(num_ftrs, masked_output_ftrs), nn.ReLU(), nn.Dropout())
        for name, param in mask_conv_model.named_parameters():
            if param.requires_grad == True:
                mask_params.append(param)
        mask_conv_model_optimizer = optim.Adam(mask_params, lr=masked_learning_rate)

        for param in other_agent_model.parameters():
            param.requires_grad = False
        num_ftrs = other_agent_model.fc.in_features
        other_agent_model.fc = nn.Sequential(nn.Linear(num_ftrs, unmasked_output_ftrs), nn.ReLU(), nn.Dropout())
        for name, param in other_agent_model.named_parameters():
            if param.requires_grad == True:
                other_agent_params.append(param)
        other_agent_model_optimizer = optim.Adam(other_agent_params, lr=unmasked_learning_rate)

        for hypo in range(num_of_hypos):
            fc_model = nn.Sequential(
                nn.Linear(4 + decoder_hidden_size, multi_fc_layer_neurons[0]), nn.ReLU(), nn.Dropout(),
                nn.Linear(multi_fc_layer_neurons[0], multi_fc_layer_neurons[1]), nn.ReLU(), nn.Dropout(),
                nn.Linear(multi_fc_layer_neurons[1], multi_fc_layer_neurons[2]), nn.ReLU(), nn.Dropout(),
                nn.Linear(multi_fc_layer_neurons[2], multi_fc_layer_neurons[3]), nn.ReLU(), nn.Dropout(),
                nn.Linear(multi_fc_layer_neurons[3], multi_fc_layer_neurons[4]), nn.ReLU(), nn.Dropout(),
                nn.Linear(multi_fc_layer_neurons[4], 4), nn.ReLU(), nn.Dropout()
            )
            fc_params = []
            for name, param in fc_model.named_parameters():
                fc_params.append(param)
            multi_hypo_optimizer = optim.Adam(fc_params, lr=multi_hypo_learning_rate)
            fc_model = fc_model.to(device)
            multi_hypo_models.append(fc_model)
            multi_hypo_params.append(fc_params)
            multi_hypo_optimizers.append(multi_hypo_optimizer)
    else:
        mask_checkpoint = torch.load(os.path.join(save_dir, mask_dir))
        epoch = mask_checkpoint['epoch']
        masked_output_ftrs = mask_checkpoint['masked_output_ftrs']
        num_ftrs = mask_conv_model.fc.in_features
        mask_conv_model.fc = nn.Linear(num_ftrs, masked_output_ftrs)
        mask_conv_model.load_state_dict(mask_checkpoint['mask_conv'])
        input_seq_len = mask_checkpoint['input_seq_len']
        target_seq_len = mask_checkpoint['target_seq_len']
        print("RECENT EPOCH WAS " + str(epoch))
        if train:
            for name, param in mask_conv_model.named_parameters():
                if param.requires_grad == True:
                    mask_params.append(param)
            mask_conv_model_optimizer = optim.Adam(mask_params, lr=masked_learning_rate)
        else:
            for param in mask_conv_model.parameters():
                param.requires_grad = False

        unmask_checkpoint = torch.load(os.path.join(save_dir, unmask_dir))
        unmasked_output_ftrs = unmask_checkpoint['unmasked_output_ftrs']
        num_ftrs = other_agent_model.fc.in_features
        other_agent_model.fc = nn.Linear(num_ftrs, unmasked_output_ftrs)
        other_agent_model.load_state_dict(unmask_checkpoint['unmask_conv'])
        if train:
            for name, param in other_agent_model.named_parameters():
                if param.requires_grad == True:
                    other_agent_params.append(param)
            other_agent_model_optimizer = optim.Adam(other_agent_params, lr=unmasked_learning_rate)
        else:
            for param in other_agent_model.parameters():
                param.requires_grad = False

        dec = torch.load(os.path.join(save_dir, dec_dir))
        decoder_hidden_size = dec['decoder_hidden_size']
        for hypo in range(num_of_hypos):
            multi_hypo_checkpoint = torch.load(os.path.join(save_dir, multi_hypo_dir_list[hypo]))
            fc_model = nn.Sequential(
                nn.Linear(4 + decoder_hidden_size, multi_fc_layer_neurons[0]), nn.ReLU(), nn.Dropout(),
                nn.Linear(multi_fc_layer_neurons[0], multi_fc_layer_neurons[1]), nn.ReLU(), nn.Dropout(),
                nn.Linear(multi_fc_layer_neurons[1], multi_fc_layer_neurons[2]), nn.ReLU(), nn.Dropout(),
                nn.Linear(multi_fc_layer_neurons[2], multi_fc_layer_neurons[3]), nn.ReLU(), nn.Dropout(),
                nn.Linear(multi_fc_layer_neurons[3], multi_fc_layer_neurons[4]), nn.ReLU(), nn.Dropout(),
                nn.Linear(multi_fc_layer_neurons[4], 4), nn.ReLU(), nn.Dropout()
            )
            fc_model.load_state_dict(multi_hypo_checkpoint['multi_hypo_' + str(hypo)]).to(device)
            multi_hypo_models.append(fc_model)
            if train:
                fc_params = []
                for name, param in fc_model.named_parameters():
                    fc_params.append(param)
                multi_hypo_optimizer = optim.Adam(fc_params, lr=multi_hypo_learning_rate)
                multi_hypo_params.append(fc_params)
                multi_hypo_optimizers.append(multi_hypo_optimizer)
            else:
                for param in fc_model.parameters():
                    param.requires_grad = False

    mask_conv_model = mask_conv_model.to(device)
    other_agent_model = other_agent_model.to(device)

    encoder = EncoderRNN(input_size=masked_output_ftrs + unmasked_output_ftrs + 4,
                         hidden_size=encoder_hidden_size, n_layers=encoder_n_layers, dropout=encoder_dropout)
    decoder = LuongAttnDecoderRNN(attn_model, input_size=4, hidden_size=decoder_hidden_size, output_size=4,
                                  n_layers=decoder_n_layers, dropout=decoder_dropout)
    cvae = CVAE(input_dim=4 + decoder_hidden_size, hidden_dim=cvae_hidden_size, latent_dim=cvae_latent_size)
    if load_model:
        enc_checkpoint = torch.load(os.path.join(save_dir, enc_dir))
        encoder_n_layers = enc_checkpoint['encoder_n_layers']
        encoder_hidden_size = enc_checkpoint['encoder_hidden_size']
        encoder_dropout = enc_checkpoint['encoder_dropout']
        encoder = EncoderRNN(input_size=masked_output_ftrs + unmasked_output_ftrs + 4,
                             hidden_size=encoder_hidden_size, n_layers=encoder_n_layers, dropout=encoder_dropout)
        encoder.load_state_dict(enc_checkpoint['encoder'])

        dec_checkpoint = torch.load(os.path.join(save_dir, dec_dir))
        decoder_n_layers = dec_checkpoint['decoder_n_layers']
        decoder_hidden_size = dec_checkpoint['decoder_hidden_size']
        decoder_dropout = dec_checkpoint['decoder_dropout']
        decoder = LuongAttnDecoderRNN(attn_model, input_size=4, hidden_size=decoder_hidden_size, output_size=4,
                                      n_layers=decoder_n_layers, dropout=decoder_dropout)
        decoder.load_state_dict(dec_checkpoint['decoder'])

        cvae_checkpoint = torch.load(cvae_dir)
        cvae_hidden_size = cvae_checkpoint['cvae_hidden_size']
        cvae_latent_size = cvae_checkpoint['cvae_latent_size']
        cvae = CVAE(input_dim=4 + decoder_hidden_size, hidden_dim=cvae_hidden_size, latent_dim=cvae_latent_size)
        cvae.load_state_dict(cvae_checkpoint['cvae'])
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    cvae = cvae.to(device)

    encoder_optimizer = []
    decoder_optimizer = []
    cvae_optimizer = []

    if train:
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=encoder_learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=decoder_learning_rate)
        cvae_optimizer = optim.Adam(cvae.parameters(), lr=cvae_learning_rate)
    else:
        for param in encoder.parameters():
            param.requires_grad = False
        for param in decoder.parameters():
            param.requires_grad = False
        for param in cvae.parameters():
            param.requires_grad = False

    loss = nn.MSELoss()

    ret_vals = {}
    ret_vals['MASKED CONV MODEL'] = mask_conv_model
    ret_vals['MASKED CONV MODEL OPTIMIZER'] = mask_conv_model_optimizer
    ret_vals['UNMASKED CONV MODEL'] = other_agent_model
    ret_vals['UNMASKED CONV MODEL OPTIMIZER'] = other_agent_model_optimizer
    ret_vals['MULTI HYPO MODELS'] = multi_hypo_models
    ret_vals['MULTI HYPO MODEL OPTIMIZERS'] = multi_hypo_optimizers
    ret_vals['ENCODER MODEL'] = encoder
    ret_vals['ENCODER MODEL OPTIMIZER'] = encoder_optimizer
    ret_vals['DECODER MODEL'] = decoder
    ret_vals['DECODER MODEL OPTIMIZER'] = decoder_optimizer
    ret_vals['CVAE MODEL'] = cvae
    ret_vals['CVAE MODEL OPTIMIZER'] = cvae_optimizer
    ret_vals['LOSS FUNC'] = loss
    ret_vals['EPOCH'] = epoch
    ret_vals['TARGET SEQ LEN'] = target_seq_len
    ret_vals['INPUT SEQ LEN'] = input_seq_len

    print("INPUT SEQUENCE LENGTH IS " + str(input_seq_len))
    print("TARGET SEQUENCE LENGTH IS " + str(target_seq_len))

    return ret_vals