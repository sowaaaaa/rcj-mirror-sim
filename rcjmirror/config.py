"""Constants and parameter containers. All lengths in millimetres.

Coordinate system: origin at field centre on the carpet, +X along the short
side, +Y along the long side (towards the blue goal), +Z up.
Field numbers come from the RCJ Soccer field specification (2026).
"""
from dataclasses import dataclass, field

# --- Field (RCJ field specification 2026) ---------------------------------
PLAY_W, PLAY_L = 1580.0, 2190.0          # playing field incl. white boundary line
OUT_AREA = 120.0                         # outer area beyond the white line
TOTAL_W, TOTAL_L = PLAY_W + 2 * OUT_AREA, PLAY_L + 2 * OUT_AREA   # 1820 x 2430
WALL_H = 220.0
WALL_T = 18.0                            # wall thickness (not specified; visual only)
LINE_W = 20.0
WEDGE_BASE, WEDGE_RISE = 100.0, 20.0
GOAL_W, GOAL_H, GOAL_D = 600.0, 100.0, 74.0
GOAL_T = 10.0                            # goal panel thickness (assumed)
PEN_W, PEN_D, PEN_R = 800.0, 250.0, 150.0   # penalty area width, depth, outer corner radius
CENTER_CIRCLE_D = 600.0
SPOT_D = 10.0
SPOT_Y = PLAY_L / 2 - 450.0              # 45 cm from the short edge
SPOT_X = PEN_W / 2                       # aligned with penalty area sides
BALL_D = 42.0

# --- Colours (RGB 0..1) -----------------------------------------------------
C_CARPET = (0.06, 0.28, 0.12)            # dark green
C_LINE = (0.92, 0.92, 0.92)
C_MARK = (0.02, 0.02, 0.02)
C_WALL = (0.03, 0.03, 0.03)
C_YELLOW = (0.95, 0.80, 0.05)
C_BLUE = (0.05, 0.25, 0.85)
C_BALL = (1.00, 0.45, 0.05)
C_ROBOT = (0.15, 0.15, 0.17)
C_BACKGROUND = (0.10, 0.10, 0.12)

# --- Robot presets (max size = diameter = height) ---------------------------
ROBOT_PRESETS = {
    "Open / Soccer Vision (180 mm)": 180.0,
    "Lightweight / Soccer Infrared (220 mm)": 220.0,
}


@dataclass
class CameraParams:
    """OpenMV Cam H7 Plus (OV5640, stock 2.8 mm M12 lens).

    hfov_deg is the stock lens value (70.8 x 55.6 deg); it is the only lens
    parameter the pinhole model needs. Change it for a different lens.
    """
    width: int = 320
    height: int = 240
    hfov_deg: float = 70.8
    # lens end of the camera, measured from the field surface
    height_above_field: float = 80.0


@dataclass
class MirrorParams:
    # distance from the lens end to the FIRST (nearest) flat plane of the mirror
    lens_to_mirror: float = 60.0
    reflectance: float = 0.88            # polished aluminium
    # mirror STEP import settings
    step_path: str = ""
    step_scale: float = 1.0              # file units -> mm (1.0 for mm files)
    flip_axis: bool = False              # use if the apex points away from the camera
    ref_plane_z: float | None = None     # override, in mirror-local mm; None = auto


@dataclass
class RobotParams:
    league: str = "Open / Soccer Vision (180 mm)"
    body_height: float = 60.0            # solid body top (below mirror)
    x: float = 0.0
    y: float = 0.0
    heading_deg: float = 0.0

    @property
    def size(self):
        return ROBOT_PRESETS[self.league]


@dataclass
class SimParams:
    camera: CameraParams = field(default_factory=CameraParams)
    mirror: MirrorParams = field(default_factory=MirrorParams)
    robot: RobotParams = field(default_factory=RobotParams)
    ball_x: float = 400.0
    ball_y: float = 300.0

    @property
    def mirror_plane_height(self):
        """Height of the mirror's first flat plane above the field."""
        return self.camera.height_above_field + self.mirror.lens_to_mirror
