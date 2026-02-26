import argparse
from dataclasses import dataclass
from typing import Optional

@dataclass
class VexEnvConfig:
    """
    Configuration object for the VEX multi-agent reinforcement learning environment.
    Holds common settings shared across training, evaluation, and simulation scripts.
    """
    game_name: str
    render_mode: Optional[str]
    experiment_path: str
    randomize: bool
    enable_communication: bool
    deterministic: bool

    @classmethod
    def add_cli_args(
        cls, 
        parser: argparse.ArgumentParser, 
        game: Optional[str] = "vexai_skills",
        render_mode: Optional[str] = "image",
        experiment_path: Optional[str] = "vex_env_output",
        randomize: Optional[bool] = True,
        communication: Optional[bool] = False,
        deterministic: Optional[bool] = False
    ) -> None:
        """
        Adds configuration arguments to an argparse.ArgumentParser.
        Defaults can be modified by passing keyword arguments. Use None to make an argument required or lack a default.
        """
        parser.add_argument(
            "--game", 
            type=str, 
            default=game,
            help="Game variant to use (e.g. 'vexai_skills', 'pushback')"
        )
        parser.add_argument(
            "--render-mode",
            type=str,
            choices=["terminal", "image", "none"],
            default=render_mode,
            help="Rendering mode: 'image' (saves frames & GIF), 'terminal' (prints text only), 'none' (silent)"
        )
        parser.add_argument(
            "--experiment-path",
            type=str,
            default=experiment_path,
            required=experiment_path is None,
            help="Output directory for logs, renders, and models"
        )
        parser.add_argument(
            "--randomize",
            action=argparse.BooleanOptionalAction,
            default=randomize,
            help="Randomize initial agent positions and orientations"
        )
        parser.add_argument(
            "--communication",
            action=argparse.BooleanOptionalAction,
            default=communication,
            help="Enable or disable agent communication"
        )
        parser.add_argument(
            "--deterministic",
            action=argparse.BooleanOptionalAction,
            default=deterministic,
            help="Enable deterministic environment mechanics (use --no-deterministic for stochastic outcomes)"
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "VexEnvConfig":
        """
        Creates a VexEnvConfig instance from parsed argparse arguments.
        """
        return cls(
            game_name=args.game,
            render_mode=args.render_mode if hasattr(args, "render_mode") and args.render_mode != "none" else None,
            experiment_path=args.experiment_path if hasattr(args, "experiment_path") else "",
            randomize=args.randomize if hasattr(args, "randomize") else True,
            enable_communication=args.communication if hasattr(args, "communication") else False,
            deterministic=args.deterministic if hasattr(args, "deterministic") else False
        )
