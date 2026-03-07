import os
import json
import argparse
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class CommunicationOption(Enum):
    NONE = "none"
    ATTENTION = "attention"
    COPY = "copy"

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
    communication_mode: CommunicationOption
    deterministic: bool

    @classmethod
    def add_cli_args(
        cls, 
        parser: argparse.ArgumentParser, 
        game: Optional[str] = "vexai_skills",
        render_mode: Optional[str] = "image",
        experiment_path: Optional[str] = "vex_env_output",
        randomize: Optional[bool] = True,
        communication_mode: Optional[str] = "none",
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
            "--communication-mode",
            type=str,
            choices=[opt.value for opt in CommunicationOption],
            default=communication_mode,
            help="Agent communication mode: 'none' (disabled), 'attention' (learned message vector), 'copy' (share observations)"
        )
        parser.add_argument(
            "--deterministic",
            action=argparse.BooleanOptionalAction,
            default=deterministic,
            help="Enable deterministic environment mechanics (use --no-deterministic for stochastic outcomes)"
        )

    @classmethod
    def read_from_metadata(cls, experiment_path: str, defaults: dict = None) -> dict:
        """Reads configuration overrides from training_metadata.json if it exists."""
        if defaults is None:
            defaults = {}
            
        if not experiment_path:
            return defaults
            
        metadata_path = os.path.join(experiment_path, "training_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                # Merge metadata into defaults
                defaults.update(metadata)
                        
                # Handle backwards compatibility with old metadata
                if "enable_communication" in metadata and "communication_mode" not in metadata:
                    defaults["communication_mode"] = CommunicationOption.ATTENTION.value if metadata["enable_communication"] else CommunicationOption.NONE.value
                        
                print(f"Loaded config overrides from metadata: {metadata_path}")
            except Exception as e:
                print(f"Warning: Could not read metadata from {metadata_path}: {e}")
                
        return defaults

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "VexEnvConfig":
        """
        Creates a VexEnvConfig instance from parsed argparse arguments,
        falling back to metadata for any unspecified arguments.
        """
        experiment_path = args.experiment_path if hasattr(args, "experiment_path") else ""
        
        # Determine defaults from arguments (only if explicitly set or if we need a base)
        defaults = {}
        if hasattr(args, "game") and args.game is not None:
             defaults["game"] = args.game
        if hasattr(args, "communication_mode") and args.communication_mode is not None:
             defaults["communication_mode"] = args.communication_mode
        if hasattr(args, "randomize") and args.randomize is not None:
             defaults["randomize"] = args.randomize
        if hasattr(args, "deterministic") and args.deterministic is not None:
             defaults["deterministic"] = args.deterministic
             
        # Read from metadata to fill in gaps or override defaults if args weren't explicitly provided
        metadata_overrides = cls.read_from_metadata(experiment_path, {})
        
        # Apply metadata ONLY if the argument was NOT explicitly provided on CLI
        # args usually have defaults, so we need to be careful. 
        # If args.game is None, we definitely use metadata.
        # If args.game has a default but metadata exists, we often want metadata to win for evaluation.
        # We will prioritize metadata over argparse defaults where possible by checking if the user explicitly provided it.
        # Since argparse doesn't easily tell us if a default was used, we will just apply metadata if it exists
        # and assume the user wants the trained model's config unless they specify otherwise.
        # A better way is to set argparse defaults to None where we want to inherit.
        
        game_name = args.game if (hasattr(args, "game") and args.game is not None) else metadata_overrides.get("game", "vexai_skills")
        communication_mode_str = args.communication_mode if (hasattr(args, "communication_mode") and args.communication_mode is not None) else metadata_overrides.get("communication_mode", "none")
        randomize = args.randomize if (hasattr(args, "randomize") and args.randomize is not None) else metadata_overrides.get("randomize", True)
        deterministic = args.deterministic if (hasattr(args, "deterministic") and args.deterministic is not None) else metadata_overrides.get("deterministic", False)

        try:
            communication_mode = CommunicationOption(communication_mode_str)
        except ValueError:
            communication_mode = CommunicationOption.NONE

        return cls(
            game_name=game_name,
            render_mode=args.render_mode if hasattr(args, "render_mode") and args.render_mode != "none" else None,
            experiment_path=experiment_path,
            randomize=randomize,
            communication_mode=communication_mode,
            deterministic=deterministic
        )
