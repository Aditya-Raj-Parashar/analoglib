"""Automatic class registry via ``__init_subclass__``.

Any concrete subclass of a registered base is auto-discovered.  Example::

    class Device(RegistryMixin, registry_name="devices"):
        ...

    class ReRAM(Device):
        ...

    Device.registry()          # {"ReRAM": <class ReRAM>}
    Device.get("ReRAM")        # <class ReRAM>
"""

from __future__ import annotations

from typing import ClassVar, Dict, Type


class RegistryMixin:
    """Mixin that auto-registers concrete subclasses.

    Usage
    -----
    Declare a base with ``registry_name``::

        class Device(RegistryMixin, registry_name="devices"):
            ...

    Every non-abstract subclass is registered by its class name.
    """

    # Each base that uses this mixin gets its own registry dict,
    # keyed by the ``registry_name`` supplied in __init_subclass_kwargs__.
    _registries: ClassVar[Dict[str, Dict[str, Type]]] = {}
    _registry_key: ClassVar[str] = ""

    def __init_subclass__(cls, registry_name: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)

        # If this subclass defines a NEW registry (i.e. it is the base):
        if registry_name is not None:
            cls._registry_key = registry_name
            cls._registries[registry_name] = {}
            return

        # Otherwise register into the parent's registry if it looks concrete
        key = cls._registry_key
        if key and not _is_abstract(cls):
            cls._registries[key][cls.__name__] = cls

    @classmethod
    def registry(cls) -> Dict[str, Type]:
        """Return the registry dict for this base's ``registry_name``."""
        return dict(cls._registries.get(cls._registry_key, {}))

    @classmethod
    def get(cls, name: str) -> Type:
        """Look up a registered subclass by name.

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        reg = cls._registries.get(cls._registry_key, {})
        if name not in reg:
            available = ", ".join(sorted(reg)) or "(none)"
            raise KeyError(
                f"No {cls._registry_key!r} registered as {name!r}. "
                f"Available: {available}"
            )
        return reg[name]


def _is_abstract(cls: Type) -> bool:
    """Return True if *cls* has unresolved abstract methods."""
    return bool(getattr(cls, "__abstractmethods__", None))
