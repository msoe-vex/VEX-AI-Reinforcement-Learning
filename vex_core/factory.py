"""
Game Adapter Factory
Instantiates game adapters dynamically based on game name.
"""

from typing import Dict, Type, List, Any
from vex_core.game_adapter import BaseGameAdapter


class GameAdapterFactory:
    _registry: Dict[str, Type[BaseGameAdapter]] = {}

    @classmethod
    def register(cls, name: str, adapter_cls: Type[BaseGameAdapter]) -> None:
        cls._registry[name.lower()] = adapter_cls

    @classmethod
    def create(cls, game_name: str, **kwargs: Any) -> BaseGameAdapter:
        name = game_name.lower()
        if name not in cls._registry:
            raise ValueError(f"Game '{game_name}' not registered. Available: {cls.get_available_games()}")
        return cls._registry[name](**kwargs)

    @classmethod
    def get_available_games(cls) -> List[str]:
        return sorted(list(cls._registry.keys()))
