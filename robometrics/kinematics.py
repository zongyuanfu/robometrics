"""Explicit, owned numerical models for the standalone CPU evaluation API."""
from dataclasses import dataclass, field
import hashlib
import json
import numpy as np

SCHEMA_VERSION = 1
_FLOAT_FIELDS = ('base', 'origins', 'axes', 'ee_offset', 'sphere_centers', 'radii',
                 'padding', 'lower', 'upper', 'velocity')
_INT_FIELDS = ('parents', 'q_indices', 'sphere_links', 'pairs')


def payload_digest(values):
    """Hash the canonical numerical model for provenance metadata."""
    digest = hashlib.sha256(f'robometrics.kinematic_model.v{SCHEMA_VERSION}\n'.encode())
    for name in _FLOAT_FIELDS + _INT_FIELDS:
        dtype = np.dtype('<f4' if name in _FLOAT_FIELDS else '<i4')
        array = np.asarray(values[name], dtype=dtype, order='C')
        digest.update(json.dumps([name, array.shape, dtype.str], separators=(',', ':')).encode()+b'\n')
        digest.update(array.tobytes())
    digest.update(json.dumps(['ee_link', int(values['ee_link'])], separators=(',', ':')).encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class KinematicModel:
    base: np.ndarray
    origins: np.ndarray
    parents: np.ndarray
    q_indices: np.ndarray
    axes: np.ndarray
    ee_link: int
    ee_offset: np.ndarray
    sphere_centers: np.ndarray
    radii: np.ndarray
    sphere_links: np.ndarray
    pairs: np.ndarray
    padding: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray
    source_fingerprint: str = ''
    payload_sha256: str = field(init=False)

    def __post_init__(self):
        for name in _FLOAT_FIELDS + _INT_FIELDS:
            raw = np.asarray(getattr(self, name))
            if not np.isfinite(raw).all():
                raise ValueError(f'{name} must be finite')
            dtype = np.dtype('<f4' if name in _FLOAT_FIELDS else '<i4')
            if name in _INT_FIELDS and (
                np.any(raw != np.floor(raw))
                or np.any(raw < -(2**31))
                or np.any(raw >= 2**31)
            ):
                raise ValueError(f'{name} must contain int32 indices')
            values = np.array(raw, dtype=dtype, order='C')
            if not np.isfinite(values).all():
                raise ValueError(f'{name} overflows float32')
            # bytes-backed arrays cannot have their write flag re-enabled.
            values = np.frombuffer(values.tobytes(), dtype=dtype).reshape(values.shape)
            object.__setattr__(self, name, values)
        if type(self.ee_link) is not int:
            raise ValueError('ee_link must be an integer')
        values = {
            name: getattr(self, name)
            for name in _FLOAT_FIELDS + _INT_FIELDS + ('ee_link',)
        }
        object.__setattr__(self, 'payload_sha256', payload_digest(values))
        # Native validation is the same for Python and direct C++ consumers.
        from . import _evaluation_cpp
        _evaluation_cpp.validate_model(self.as_mapping())

    @property
    def dof(self):
        return len(self.lower)

    def as_mapping(self):
        return {name:getattr(self, name) for name in self.__dataclass_fields__}
