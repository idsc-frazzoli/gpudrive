from dataclasses import dataclass

from networks.perm_eq_late_fusion import LateFusionNet, LateFusionPolicy


@dataclass
class ExperimentConfig:
    """Configurations for experiments."""

    # General
    device: str = "cuda"

    # Dataset
    data_dir: str = "waymo_data"

    # Logging
    use_wandb: bool = True
    sync_tensorboard: bool = True
    logging_collection_window: int = (
        1000  # how many trajectories we average logs over
    )
    log_freq: int = 100

    # Hyperparameters
    seed: int = 42
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    n_steps: int = 92  # Has to be at least > episode_length = 91
    batch_size: int = 2048
    verbose: int = 0
    total_timesteps: int = 10e7

    # Network
    mlp_class = LateFusionNet
    policy = LateFusionPolicy
    ego_state_layers = [64, 32]
    road_object_layers = [64, 64]
    road_graph_layers = [64, 64]
    shared_layers = [64, 64]
    act_func = "tanh"
    dropout = 0.0
    last_layer_dim_pi = 64
    last_layer_dim_vf = 64

    # Wandb
    project_name = "gpudrive"
    group_name = "dc/PPO"
    entity = "_emerge"
