import os
import json
import argparse
import numpy as np
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
    temperature: float = 1.0
    copy_message_dropout_prob: float = 0.0

    @classmethod
    def add_cli_args(
        cls, 
        parser: argparse.ArgumentParser, 
        game: Optional[str] = "vexai_skills",
        render_mode: Optional[str] = "image",
        experiment_path: Optional[str] = "vex_env_output",
        randomize: Optional[bool] = True,
        communication_mode: Optional[str] = "none",
        deterministic: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        copy_message_dropout_prob: Optional[float] = 0.0,
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
        parser.add_argument(
            "--temperature",
            type=float,
            default=temperature,
            help="Action selection temperature (0 < T). Lower => more deterministic. Default=1.0",
        )
        parser.add_argument(
            "--copy-message-dropout-prob",
            type=float,
            default=copy_message_dropout_prob,
            help="In COPY communication mode, probability [0,1] of zeroing each teammate message per tick.",
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

        # Collect explicit CLI-provided values (so metadata can override defaults when appropriate)
        defaults = {}
        if hasattr(args, "game") and args.game is not None:
            defaults["game"] = args.game
        if hasattr(args, "communication_mode") and args.communication_mode is not None:
            defaults["communication_mode"] = args.communication_mode
        if hasattr(args, "randomize") and args.randomize is not None:
            defaults["randomize"] = args.randomize
        if hasattr(args, "deterministic") and args.deterministic is not None:
            defaults["deterministic"] = args.deterministic
        if hasattr(args, "temperature") and args.temperature is not None:
            defaults["temperature"] = args.temperature
        if hasattr(args, "copy_message_dropout_prob") and args.copy_message_dropout_prob is not None:
            defaults["copy_message_dropout_prob"] = args.copy_message_dropout_prob

        # Read metadata overrides from the experiment directory (if present)
        metadata_overrides = cls.read_from_metadata(experiment_path, {})

        # Resolve final values: prefer CLI args, then metadata, then hardcoded defaults
        game_name = args.game if (hasattr(args, "game") and args.game is not None) else metadata_overrides.get("game", "vexai_skills")
        communication_mode_str = args.communication_mode if (hasattr(args, "communication_mode") and args.communication_mode is not None) else metadata_overrides.get("communication_mode", "none")
        randomize = args.randomize if (hasattr(args, "randomize") and args.randomize is not None) else metadata_overrides.get("randomize", True)
        deterministic = args.deterministic if (hasattr(args, "deterministic") and args.deterministic is not None) else metadata_overrides.get("deterministic", False)
        temperature = args.temperature if (hasattr(args, "temperature") and args.temperature is not None) else metadata_overrides.get("temperature", 1.0)
        copy_message_dropout_prob = args.copy_message_dropout_prob if (hasattr(args, "copy_message_dropout_prob") and args.copy_message_dropout_prob is not None) else metadata_overrides.get("copy_message_dropout_prob", 0.0)

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
            deterministic=deterministic,
            temperature=float(temperature),
            copy_message_dropout_prob=float(np.clip(copy_message_dropout_prob, 0.0, 1.0)),
        )
