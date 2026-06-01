"""Baseline registry for MaTBridge experiments."""

from matbridge.baselines.bridge_off import BridgeOffBaseline
from matbridge.baselines.copula_gaussian import CopulaCalibGaussianJoint
from matbridge.baselines.copula_vine import CopulaCalibVineJoint
from matbridge.baselines.denseloss import DenseLossBaseline
from matbridge.baselines.erm import ERMBaseline
from matbridge.baselines.fds import FDSBaseline
from matbridge.baselines.smoter import SMOTERBaseline
from matbridge.baselines.tabddpm_aug import TabDDPMAugBaseline
from matbridge.baselines.tabdiff_aug import TabDiffAugBaseline

BASELINE_REGISTRY = {
    "erm": ERMBaseline,
    "fds": FDSBaseline,
    "denseloss": DenseLossBaseline,
    "copula_gaussian_joint": CopulaCalibGaussianJoint,
    "copula_vine_joint": CopulaCalibVineJoint,
    "bridge_off": BridgeOffBaseline,
    "tabddpm_aug": TabDDPMAugBaseline,
    "tabdiff_aug": TabDiffAugBaseline,
    "smoter": SMOTERBaseline,
}

__all__ = ["BASELINE_REGISTRY"]
