"""
AUCell scoring of region signatures on imputed accessibility rankings.

Reimplementation of ``ctxcore.aucell.aucell4r`` (Van de Sande et al., 2020) for
signatures with uniform weights, which gives identical AUC values without depending
on ctxcore.
"""

from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt


def derive_rank_cutoff(auc_threshold: float, total_regions: int) -> int:
    """
    Get the (0-based) rank cutoff used for calculating the AUC.

    Parameters
    ----------
    auc_threshold
        The fraction of the ranked regions to take into account for the calculation
        of the Area Under the recovery Curve.
    total_regions
        The total number of ranked regions.

    Returns
    -------
    Rank cutoff. Only ranks strictly smaller than this value contribute to the AUC.

    """
    if not 0.0 < auc_threshold <= 1.0:
        raise ValueError(
            f"AUC threshold ({auc_threshold}) must be a fraction between 0.0 and 1.0."
        )

    # In the R implementation the cutoff is rounded.
    rank_cutoff = round(auc_threshold * total_regions)

    if not 0 < rank_cutoff <= total_regions - 1:
        raise ValueError(
            f"An AUC threshold of {auc_threshold} corresponds to {rank_cutoff} top "
            f"ranked regions out of {total_regions}. Increase the AUC threshold or "
            "the number of ranked regions."
        )

    # Make sure we have exactly the same AUC values as the R-SCENIC pipeline.
    # In the latter the rank threshold is not included in AUC calculation.
    return rank_cutoff - 1


@numba.njit(parallel=True)
def _aucs_uniform_weights(
    rankings: npt.NDArray[np.int32],
    signature_region_indices: npt.NDArray[np.intp],
    rank_cutoff: int,
    max_auc: float,
) -> npt.NDArray[np.float64]:
    # The recovery curve of a cell steps up by 1 at each signature region ranked
    # below rank_cutoff, so its area up to rank_cutoff is the sum over those regions
    # of (rank_cutoff - rank).
    n_cells = rankings.shape[1]
    aucs = np.zeros(n_cells, dtype=np.float64)
    for cell_idx in numba.prange(n_cells):
        auc = 0
        for region_idx in signature_region_indices:
            rank = rankings[region_idx, cell_idx]
            if rank < rank_cutoff:
                auc += rank_cutoff - rank
        aucs[cell_idx] = auc / max_auc
    return aucs


def aucell(
    rankings: npt.NDArray[np.int32],
    region_names: npt.ArrayLike,
    signatures: dict[str, npt.ArrayLike],
    auc_threshold: float = 0.05,
    min_frac_signature_present: float = 0.80,
) -> npt.NDArray[np.float64]:
    """
    Calculate AUCell enrichment of region signatures per cell.

    Parameters
    ----------
    rankings
        0-based rankings per cell (regions x cells): the most accessible region
        of a cell has rank 0.
    region_names
        Names of the ranked regions (same order as the rows of ``rankings``).
    signatures
        Dictionary with signature names as keys and region names as values.
    auc_threshold
        The fraction of the ranked regions to take into account for the calculation
        of the Area Under the recovery Curve.
    min_frac_signature_present
        Minimum fraction of (unique) signature regions that needs to be present in
        ``region_names``. Signatures below this fraction get an AUC of 0 for all
        cells.

    Returns
    -------
    Numpy array with AUC values (cells x signatures).

    """
    region_names = np.asarray(region_names)

    if rankings.shape[0] != region_names.shape[0]:
        raise ValueError(
            f"Number of region names ({region_names.shape[0]}) does not match the "
            f"number of rows of rankings ({rankings.shape[0]})."
        )

    rank_cutoff = derive_rank_cutoff(auc_threshold, rankings.shape[0])

    aucs = np.zeros((rankings.shape[1], len(signatures)), dtype=np.float64)
    for signature_idx, signature_region_names in enumerate(signatures.values()):
        signature_region_names = np.unique(np.asarray(signature_region_names))
        signature_region_indices = np.flatnonzero(
            np.isin(region_names, signature_region_names)
        )

        if (
            signature_region_indices.shape[0] == 0
            or signature_region_indices.shape[0] / signature_region_names.shape[0]
            < min_frac_signature_present
        ):
            continue

        # For reason of generating the same results as in R, one is added to the
        # rank cutoff for calculating the maximum AUC.
        max_auc = float((rank_cutoff + 1) * signature_region_indices.shape[0])

        aucs[:, signature_idx] = _aucs_uniform_weights(
            rankings, signature_region_indices, rank_cutoff, max_auc
        )

    return aucs
