"""Detector plugins. Import submodules only via register_all() to avoid cycles."""


def register_all() -> None:
    from veilscan.detectors.spatial import register_spatial
    from veilscan.detectors.frequency import register_frequency
    from veilscan.detectors.residual import register_residual
    from veilscan.detectors.deep import register_deep
    from veilscan.detectors.blackbox import register_blackbox
    from veilscan.detectors.foundation import register_foundation
    from veilscan.detectors.latent import register_latent

    register_spatial()
    register_frequency()
    register_residual()
    register_deep()
    register_blackbox()
    register_foundation()
    register_latent()
