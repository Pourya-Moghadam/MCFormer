"""Training-only pose, object-track, trajectory, and coupling preprocessing."""

from mcformer.auxiliary.cache import ObservationBundle, ObservationCache
from mcformer.auxiliary.coupling import CouplingTarget, compute_coupling_target
from mcformer.auxiliary.pipeline import SampleTarget, TargetSettings, build_sample_target
from mcformer.auxiliary.trajectories import ObjectTrajectory, PositionTrajectory

__all__ = [
    "CouplingTarget",
    "ObservationBundle",
    "ObservationCache",
    "ObjectTrajectory",
    "PositionTrajectory",
    "SampleTarget",
    "TargetSettings",
    "build_sample_target",
    "compute_coupling_target",
]
