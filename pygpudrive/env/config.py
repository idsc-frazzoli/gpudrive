"""Configuration classes and enums for GPUDrive Environments."""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional
import torch

import gpudrive
from pygpudrive.env import constants

@dataclass
class EnvConfig:
    """Configuration settings for the GPUDrive gym environment.

    This class contains both Python-specific configurations and settings that
    are shared between Python and C++ components of the simulator. 

    To modify simulator settings shared with C++, follow these steps:
    1. Navigate to `src/consts.hpp` in the C++ codebase.
    2. Locate and modify the desired constant definitions (e.g., `kMaxAgentCount`).
    3. Save the changes to `src/consts.hpp`.
    4. Recompile the simulator to apply changes across both C++ and Python environments.
    """

    # Python-specific configurations
    # Observation space settings
    ego_state: bool = True  # Include ego vehicle state in observations
    road_map_obs: bool = True  # Include road graph in observations
    partner_obs: bool = True  # Include partner vehicle info in observations
    norm_obs: bool = True  # Normalize observations
    enable_lidar: bool = False  # Enable LiDAR data in observations
    
    # Set the weights for the reward components
    # R = a * collided + b * goal_achieved + c * off_road
    collision_weight: float = 0.0
    goal_achieved_weight: float = 1.0
    off_road_weight: float = 0.0

    # Road observation algorithm settings
    road_obs_algorithm: str = "linear"  # Algorithm for road observations
    obs_radius: float = 100.0  # Radius for road observations
    polyline_reduction_threshold: float = 1.0  # Threshold for polyline reduction

    # Action space settings (joint discrete)
    steer_actions: torch.Tensor = torch.round(torch.linspace(-1.0, 1.0, 13), decimals=3)
    accel_actions: torch.Tensor = torch.round(torch.linspace(-4.0, 4.0, 7), decimals=3)

    # Collision behavior settings
    collision_behavior: str = "remove"  # Options: "remove", "stop", "ignore"

    # Scene configuration
    remove_non_vehicles: bool = True  # Remove non-vehicle entities from the scene

    # Reward
    reward_type: str = "weighted_combination" # options: "sparse_on_goal_achieved" / "weighted_combination"
    
    # The radius around the goal point within which the agent is considered
    # to have reached the goal
    dist_to_goal_threshold: float = 3.0

    # Maximum number of controlled vehicles and feature dimensions for network
    MAX_CONTROLLED_VEHICLES: int = 128
    ROADMAP_AGENT_FEAT_DIM: int = MAX_CONTROLLED_VEHICLES - 1
    TOP_K_ROADPOINTS: int = (
        200  # Number of visible roadpoints from the road graph
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # VALUES BELOW ARE ENV CONSTANTS: DO NOT CHANGE # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Normalization constants
    max_speed: int = 100
    max_veh_len: int = 30
    max_veh_width: int = 10
    min_rel_goal_coord: int = -1000
    max_rel_goal_coord: int = 1000
    min_rel_agent_pos: int = -1000
    max_rel_agent_pos: int = 1000
    max_orientation_rad: float = 2 * np.pi
    min_rm_coord: int = -1000
    max_rm_coord: int = 1000
    max_road_line_segmment_len: int = 100
    max_road_scale: int = 100

    # Feature dimensions
    EGO_STATE_DIM = 6 if ego_state else 0
    PARTNER_DIM = 10 if partner_obs else 0
    ROAD_MAP_DIM = 13 if road_map_obs else 0
    
    # Agent tensor shape set in consts.hpp (kMaxAgentCount)
    k_max_agent_count: int = 128
    
class SelectionDiscipline(Enum):
    """Enum for selecting scenes discipline in dataset configuration."""
    FIRST_N = 0
    RANDOM_N = 1
    PAD_N = 2
    EXACT_N = 3
    K_UNIQUE_N = 4


@dataclass
class SceneConfig:
    """Configuration for selecting scenes from a dataset.

    Attributes:
        path (str): Path to the dataset.
        num_scenes (int): Number of scenes to select.
        discipline (SelectionDiscipline): Method for selecting scenes.
        k_unique_scenes (Optional[int]): Number of unique scenes if using K_UNIQUE_N discipline.
    """
    path: str
    num_scenes: int
    discipline: SelectionDiscipline = SelectionDiscipline.PAD_N
    k_unique_scenes: Optional[int] = None


class RenderMode(Enum):
    """Enum for specifying rendering mode."""
    PYGAME_ABSOLUTE = "pygame_absolute"
    PYGAME_EGOCENTRIC = "pygame_egocentric"
    PYGAME_LIDAR = "pygame_lidar"
    MADRONA_RGB = "madrona_rgb"
    MADRONA_DEPTH = "madrona_depth"


class PygameOption(Enum):
    """Enum for Pygame rendering options."""
    HUMAN = "human"
    RGB = "rgb"


class MadronaOption(Enum):
    """Enum for Madrona rendering options."""
    AGENT_VIEW = "agent_view"
    TOP_DOWN = "top_down"


@dataclass
class RenderConfig:
    """Configuration settings for rendering the environment.

    Attributes:
        render_mode (RenderMode): The mode used for rendering the environment.
        view_option (Enum): Specific rendering view option (e.g., RGB, human view).
        resolution (Tuple[int, int]): Resolution of the rendered image.
        line_thickness (int): Thickness of the road lines in the rendering.
        draw_obj_idx (bool): Whether to draw object indices on objects.
        obj_idx_font_size (int): Font size for object indices.
        color_scheme (str): Color scheme for the rendering ("light" or "dark").
    """
    render_mode: RenderMode = RenderMode.PYGAME_ABSOLUTE
    view_option: Enum = PygameOption.RGB
    resolution: Tuple[int, int] = (1024, 1024)
    line_thickness: int = 0.7
    draw_obj_idx: bool = False
    obj_idx_font_size: int = 9
    color_scheme: str = "light"

    def __str__(self) -> str:
        """Returns a string representation of the rendering configuration."""
        return (f"RenderMode: {self.render_mode.value}, ViewOption: {self.view_option.value}, "
                f"Resolution: {self.resolution}, LineThickness: {self.line_thickness}, "
                f"DrawObjectIdx: {self.draw_obj_idx}, ObjectIdxFontSize: {self.obj_idx_font_size}, "
                f"ColorScheme: {self.color_scheme}")
