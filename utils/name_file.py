def name_file_start(dict):
    str_add = '_'.join([str(val_i) + str_i for str_i, val_i in dict.items()])
    file_x = 'x_' + str_add
    file_y = 'y_' + str_add
    return file_x, file_y

def name_synthetic_dict(dict):
    keys = ['obj', 'loss', 'proj_dim', 'input_dim', 'quantile']
    name_file = 'Synthetic'
    for key_i in keys:
        name_file += '_' + key_i + str(dict[key_i])
    return name_file

def name_dict_wo_quantile(dict):
    keys = ['obj', 'loss', 'proj_dim', 'input_dim']
    name_file = 'Sensitivity'
    for key_i in keys:
        name_file += '_' + key_i + str(dict[key_i])
    return name_file

def name_synthetic_dict_no_quantile(dict):
    keys = ['obj', 'loss', 'proj_dim', 'input_dim']
    name_file = 'Synthetic'
    for key_i in keys:
        name_file += '_' + key_i + str(dict[key_i])
    return name_file

def name_synthetic(dict):
    keys = ['obj', 'opt', 'loss', 'proj_dim', 'input_dim']
    name_file = 'Synthetic'
    for key_i in keys:
        name_file += '_' + key_i + str(dict[key_i])
    return name_file

def name_model_vae(dict):
    keys = ['obj', 'opt', 'proj_dim', 'input_dim']
    name_file = 'Synthetic'
    for key_i in keys:
        name_file += '_' + key_i + str(dict[key_i])
    return name_file

def name_turbo():
    from examples.config_file import config
    ig = config()
    if ig.uneven_terrain:
       name_file = 'turbo_'  + str(ig.n_trust_regions) + 'bs_' + str(ig.batch_size) + 'n_eval' + str(ig.max_evals) + \
                   'motion_' + ig.motion + 'uneven_'
    else:
       name_file = 'turbo_'  + str(ig.n_trust_regions) + 'bs_' + str(ig.batch_size) + 'n_eval' + str(ig.max_evals) + \
                   'motion_' + ig.motion + 'even_'
    name_file += 'ae_type_' + ig.ea_type
    if ig.ea_type != 'no':
       name_file += 'hd_' + str(ig.high_dim) + 'ld_' + str(ig.low_dim)
    return name_file


