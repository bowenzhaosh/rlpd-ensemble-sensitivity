from configs import sac_config
from ml_collections.config_dict import config_dict
def get_config():
    config = sac_config.get_config()
    config.num_qs = 10
    config.num_min_qs = 2
    config.critic_layer_norm=True
    config.spec_norm_coef = config_dict.placeholder(float)
    return config
