from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy.stats import gaussian_kde
from tss_profile_reference import reference_tss_profile

from pycisTopic.categoricals import PycisTopicCategoricals
from pycisTopic.qc import (
    compute_kde,
    compute_qc_stats,
    get_barcodes_passing_qc_for_sample,
    get_otsu_threshold,
)
from pycisTopic.tss_profile import get_tss_profile

CHROM_SIZE = 2_000_000
TSS_POSITIONS = np.arange(20_000, CHROM_SIZE - 20_000, 20_000)


@pytest.fixture(scope="module")
def tss_annotation() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Chromosome": ["chr1"] * len(TSS_POSITIONS),
            "TSS_start": TSS_POSITIONS.astype(np.int32),
            "Strand": ["+", "-"] * (len(TSS_POSITIONS) // 2)
            + ["+"] * (len(TSS_POSITIONS) % 2),
        }
    )


@pytest.fixture(scope="module")
def peaks() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Chromosome": ["chr1"] * len(TSS_POSITIONS),
            "Start": (TSS_POSITIONS - 500).astype(np.int32),
            "End": (TSS_POSITIONS + 500).astype(np.int32),
        }
    )


@pytest.fixture(scope="module")
def fragments() -> pl.DataFrame:
    """Good cells: many fragments around TSSs. Bad cells: few uniform fragments."""
    rng = np.random.default_rng(0)
    rows = []
    for cell_idx in range(30):
        n_near = rng.integers(400, 800)
        starts = rng.choice(TSS_POSITIONS, n_near) + rng.integers(-150, 150, n_near)
        n_uniform = rng.integers(50, 100)
        starts = np.concatenate([starts, rng.integers(0, CHROM_SIZE - 500, n_uniform)])
        rows.append((f"good{cell_idx}", starts))
    for cell_idx in range(30):
        n_uniform = rng.integers(30, 80)
        rows.append((f"bad{cell_idx}", rng.integers(0, CHROM_SIZE - 500, n_uniform)))
    chroms, all_starts, ends, cbs = [], [], [], []
    for cb, starts in rows:
        lengths = rng.integers(40, 400, len(starts))
        chroms += ["chr1"] * len(starts)
        all_starts.append(starts)
        ends.append(starts + lengths)
        cbs += [cb] * len(starts)
    return pl.DataFrame(
        {
            "Chromosome": pl.Series(chroms, dtype=pl.Categorical(PycisTopicCategoricals.CHROMOSOME)),
            "Start": np.concatenate(all_starts).astype(np.int32),
            "End": np.concatenate(ends).astype(np.int32),
            "CB": pl.Series(cbs, dtype=pl.Categorical(PycisTopicCategoricals.CB)),
            "CB_count": np.ones(len(cbs), dtype=np.int32),
        }
    )


class TestTssProfile:
    def test_cut_site_positions_are_strand_aware(self):
        tss = pl.DataFrame(
            {"Chromosome": ["chr1", "chr1"], "TSS_start": [10_000, 50_000], "Strand": ["+", "-"]}
        )
        fragments = pl.DataFrame(
            {
                "Chromosome": pl.Series(["chr1", "chr1"], dtype=pl.Categorical(PycisTopicCategoricals.CHROMOSOME)),
                "Start": pl.Series([10_100, 49_800], dtype=pl.Int32),
                "End": pl.Series([10_200, 49_900], dtype=pl.Int32),
                "CB": pl.Series(["A", "B"], dtype=pl.Categorical(PycisTopicCategoricals.CB)),
            }
        )
        _, _, per_cb = get_tss_profile(
            fragments,
            tss,
            flank_window=300,
            smoothing_rolling_window=1,
            min_norm=1.0,
            tss_norm_matrix_per_cb_cbs=pl.Series(["A", "B"]),
        )
        counts = {
            cb: per_cb.filter(pl.col(cb) > 0)["position_from_tss"].to_list()
            for cb in ("A", "B")
        }
        # + strand: fragment start and end - 1 relative to the TSS.
        assert counts["A"] == [100, 199]
        # - strand: positions are mirrored around the TSS.
        assert counts["B"] == [101, 200]

    def test_profile_shape_and_enrichment(self, fragments, tss_annotation):
        enrichment, sample_profile, per_cb = get_tss_profile(
            fragments, tss_annotation, flank_window=1000
        )
        assert sample_profile.height == 2001
        assert sample_profile["position_from_tss"].to_list()[0] == -1000
        assert per_cb is None
        tss_enrichment = dict(
            zip(enrichment["CB"].cast(pl.Utf8), enrichment["tss_enrichment"])
        )
        good = np.mean([v for k, v in tss_enrichment.items() if k.startswith("good")])
        bad = np.mean([v for k, v in tss_enrichment.items() if k.startswith("bad")])
        assert good > 3 * bad

    @pytest.mark.parametrize(
        ("flank_window", "smoothing_rolling_window", "minimum_signal_window", "tss_window", "min_norm"),
        [(1000, 10, 100, 50, 0.2), (300, 1, 20, 0, 1.0), (500, 25, 50, 100, 0.01)],
    )
    def test_matches_reference(
        self,
        fragments,
        tss_annotation,
        flank_window,
        smoothing_rolling_window,
        minimum_signal_window,
        tss_window,
        min_norm,
    ):
        # Add a duplicated TSS and a TSS close to the start of the chromosome.
        tss_annotation = pl.concat(
            [
                tss_annotation,
                tss_annotation.head(3),
                pl.DataFrame(
                    {"Chromosome": ["chr1"], "TSS_start": [5], "Strand": ["-"]},
                    schema=tss_annotation.schema,
                ),
            ]
        )
        fragments = pl.concat(
            [
                fragments,
                pl.DataFrame(
                    {
                        "Chromosome": pl.Series(["chr1"], dtype=fragments["Chromosome"].dtype),
                        "Start": pl.Series([0], dtype=pl.Int32),
                        "End": pl.Series([60], dtype=pl.Int32),
                        "CB": pl.Series(["good0"], dtype=fragments["CB"].dtype),
                        "CB_count": pl.Series([1], dtype=pl.Int32),
                    }
                ),
            ]
        )
        kwargs = {
            "flank_window": flank_window,
            "smoothing_rolling_window": smoothing_rolling_window,
            "minimum_signal_window": minimum_signal_window,
            "tss_window": tss_window,
            "min_norm": min_norm,
        }
        expected_enrichment, expected_sample, expected_per_cb = reference_tss_profile(
            fragments, tss_annotation, **kwargs
        )
        cbs = expected_per_cb.columns[1:]
        enrichment, sample, per_cb = get_tss_profile(
            fragments,
            tss_annotation,
            tss_norm_matrix_per_cb_cbs=pl.Series(cbs),
            no_threads=3,
            **kwargs,
        )

        expected_enrichment = expected_enrichment.sort("CB")
        enrichment = enrichment.sort("CB")
        assert enrichment["CB"].cast(pl.Utf8).equals(expected_enrichment["CB"].cast(pl.Utf8))
        np.testing.assert_allclose(
            enrichment["tss_enrichment"], expected_enrichment["tss_enrichment"], rtol=1e-12
        )
        assert sample["position_from_tss"].equals(expected_sample["position_from_tss"])
        np.testing.assert_allclose(
            sample["normalized_tss_enrichment"],
            expected_sample["normalized_tss_enrichment"],
            rtol=1e-12,
        )
        assert per_cb.columns == expected_per_cb.columns
        np.testing.assert_allclose(
            per_cb.to_numpy(), expected_per_cb.to_numpy(), rtol=1e-12, atol=1e-12
        )

    def test_string_columns(self, fragments, tss_annotation):
        # Chromosome and CB columns as strings instead of categoricals.
        expected = get_tss_profile(fragments, tss_annotation, flank_window=1000)
        result = get_tss_profile(
            fragments.with_columns(pl.col("Chromosome", "CB").cast(pl.Utf8)),
            tss_annotation,
            flank_window=1000,
        )
        assert result[0].sort("CB").equals(expected[0].sort("CB"))
        assert result[1].equals(expected[1])

    def test_no_overlap(self, fragments):
        tss = pl.DataFrame({"Chromosome": ["chrZ"], "TSS_start": [100], "Strand": ["+"]})
        with pytest.raises(ValueError, match="same chromosome name source"):
            get_tss_profile(fragments, tss)


def test_compute_kde_matches_scipy():
    rng = np.random.default_rng(1)
    training = rng.normal(size=(2, 200))
    test = np.hstack([training[:, :50], training[:, :10]])  # with duplicates
    np.testing.assert_allclose(
        compute_kde(training, test, no_threads=3), gaussian_kde(training)(test)
    )


def test_compute_kde_binned_approximates_scipy():
    rng = np.random.default_rng(2)
    # Correlated, skewed data on different scales (like QC statistics).
    x = rng.gamma(2.0, 1.0, size=20_000)
    training = np.vstack([x, 10 * x + rng.normal(0, 5, size=x.shape[0])])
    test = training[:, :500]
    expected = gaussian_kde(training)(test)
    result = compute_kde(training, test, method="binned")
    not_negligible = expected > 1e-3 * expected.max()
    np.testing.assert_allclose(
        result[not_negligible], expected[not_negligible], rtol=1e-2
    )
    assert np.abs(result - expected).max() < 1e-2 * expected.max()


@pytest.mark.parametrize("method", ["exact", "binned"])
def test_compute_kde_constant_second_dimension(method):
    training = np.vstack([np.arange(10.0), np.zeros(10)])
    # The constant dimension is ignored.
    np.testing.assert_allclose(
        compute_kde(training, training, method=method),
        gaussian_kde(training[0])(training[0]),
        rtol=1e-2 if method == "binned" else 1e-12,
    )


@pytest.fixture(scope="module")
def qc_stats(fragments, peaks, tss_annotation):
    return compute_qc_stats(
        fragments_df_pl=fragments,
        regions_df_pl=peaks,
        tss_annotation=tss_annotation,
        tss_flank_window=1000,
        min_fragments_per_cb=10,
        no_threads=2,
    )


class TestComputeQcStats:
    def test_outputs(self, qc_stats, fragments):
        stats, insert_sizes, tss_sample, tss_per_cb = qc_stats
        assert stats.height == fragments["CB"].n_unique()
        for column in (
            "barcode_rank",
            "unique_fragments_in_peaks_count",
            "fraction_of_fragments_in_peaks",
            "tss_enrichment",
            "pdf_values_for_tss_enrichment",
            "pdf_values_for_fraction_of_fragments_in_peaks",
            "pdf_values_for_duplication_ratio",
        ):
            assert column in stats.columns
        assert stats["fraction_of_fragments_in_peaks"].is_between(0, 1).all()
        assert stats["pdf_values_for_tss_enrichment"].is_finite().all()
        assert insert_sizes["fragments_count"].sum() == fragments.height
        assert tss_sample.height == 2001
        assert tss_per_cb is None

    def test_good_cells_score_higher(self, qc_stats):
        stats = qc_stats[0].with_columns(
            pl.col("CB").cast(pl.Utf8).str.starts_with("good").alias("good")
        )
        means = stats.group_by("good").agg(
            pl.col("fraction_of_fragments_in_peaks").mean(),
            pl.col("tss_enrichment").mean(),
        )
        good = means.filter(pl.col("good")).row(0, named=True)
        bad = means.filter(~pl.col("good")).row(0, named=True)
        assert good["fraction_of_fragments_in_peaks"] > 0.5 > bad["fraction_of_fragments_in_peaks"]
        assert good["tss_enrichment"] > bad["tss_enrichment"]

    def test_otsu_thresholds_separate_good_and_bad_cells(self, qc_stats):
        fragments_threshold, tss_threshold, passing = get_otsu_threshold(
            qc_stats[0], min_otsu_fragments=1, min_otsu_tss=0.0
        )
        passing_cbs = set(passing["CB"].cast(pl.Utf8).to_list())
        assert passing_cbs == {f"good{i}" for i in range(30)}
        assert fragments_threshold > 0
        assert tss_threshold > 0

    def test_get_barcodes_passing_qc_for_sample(self, qc_stats, tmp_path):
        stats = qc_stats[0]
        stats.write_parquet(tmp_path / "s1.fragments_stats_per_cb.parquet")
        cbs, thresholds = get_barcodes_passing_qc_for_sample(
            "s1", tmp_path, unique_fragments_threshold=0, use_automatic_thresholds=False
        )
        assert len(cbs) == stats.height
        assert thresholds == {
            "unique_fragments_threshold": 0,
            "tss_enrichment_threshold": 0.0,
            "frip_threshold": 0.0,
        }

        fragments_threshold, tss_threshold, _ = get_otsu_threshold(
            stats, min_otsu_fragments=1, min_otsu_tss=0.0
        )
        pl.DataFrame(
            {
                "unique_fragments_in_peaks_count_otsu_threshold": [fragments_threshold],
                "tss_enrichment_otsu_threshold": [tss_threshold],
            }
        ).write_csv(tmp_path / "s1.otsu_thresholds.tsv", separator="\t")
        cbs, thresholds = get_barcodes_passing_qc_for_sample("s1", tmp_path)
        assert all(cb.startswith("good") for cb in cbs)
        assert thresholds["tss_enrichment_threshold"] == pytest.approx(tss_threshold)

    def test_missing_stats_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_barcodes_passing_qc_for_sample(
                "missing", tmp_path, use_automatic_thresholds=False
            )
