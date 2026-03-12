"""Tests for TPC-DS query generation and data loading."""

import json
import tempfile
from pathlib import Path

import pytest

from provenance_qs.data.tpcds import (
    TPCDS_TEMPLATES,
    generate_query_instances,
    load_query_dataset,
)
from provenance_qs.data.loader import (
    ProvenanceRecord,
    ProvenanceDataset,
    save_provenance_jsonl,
    load_provenance_jsonl,
)


# ---------------------------------------------------------------------------
# TPCDS_TEMPLATES
# ---------------------------------------------------------------------------

class TestTPCDSTemplates:
    def test_nonempty(self):
        assert len(TPCDS_TEMPLATES) > 0

    def test_has_required_keys(self):
        for tid, template in TPCDS_TEMPLATES.items():
            assert "sql" in template, f"Template {tid} missing 'sql'"
            assert "params" in template, f"Template {tid} missing 'params'"
            assert "description" in template, f"Template {tid} missing 'description'"

    def test_sql_is_string(self):
        for tid, template in TPCDS_TEMPLATES.items():
            assert isinstance(template["sql"], str)
            assert len(template["sql"]) > 10


# ---------------------------------------------------------------------------
# generate_query_instances
# ---------------------------------------------------------------------------

class TestGenerateQueryInstances:
    def test_returns_correct_count(self):
        tid = next(iter(TPCDS_TEMPLATES))
        instances = generate_query_instances(tid, n_instances=5)
        assert len(instances) == 5

    def test_returns_correct_template_ids(self):
        tid = next(iter(TPCDS_TEMPLATES))
        instances = generate_query_instances(tid, n_instances=5)
        for sql, label in instances:
            assert label == tid

    def test_sql_is_non_empty(self):
        tid = next(iter(TPCDS_TEMPLATES))
        instances = generate_query_instances(tid, n_instances=3)
        for sql, _ in instances:
            assert isinstance(sql, str) and len(sql) > 0

    def test_reproducible_with_seed(self):
        import random
        tid = next(iter(TPCDS_TEMPLATES))
        rng1 = random.Random(99)
        rng2 = random.Random(99)
        a = generate_query_instances(tid, n_instances=5, rng=rng1)
        b = generate_query_instances(tid, n_instances=5, rng=rng2)
        assert a == b

    def test_different_seeds_may_differ(self):
        import random
        tid = next(iter(TPCDS_TEMPLATES))
        # With enough instances it's highly unlikely all are identical
        instances = generate_query_instances(tid, n_instances=50)
        sqls = [sql for sql, _ in instances]
        # There should be at least 2 distinct SQL strings
        assert len(set(sqls)) > 1

    def test_unknown_template_raises(self):
        with pytest.raises(ValueError, match="not found"):
            generate_query_instances(9999, n_instances=1)


# ---------------------------------------------------------------------------
# load_query_dataset
# ---------------------------------------------------------------------------

class TestLoadQueryDataset:
    def test_length(self):
        n = 10
        queries, labels = load_query_dataset(n_instances_per_template=n)
        assert len(queries) == n * len(TPCDS_TEMPLATES)
        assert len(labels) == len(queries)

    def test_labels_are_valid_template_ids(self):
        _, labels = load_query_dataset(n_instances_per_template=5)
        for label in labels:
            assert label in TPCDS_TEMPLATES

    def test_subset_of_templates(self):
        tids = [1, 3]
        queries, labels = load_query_dataset(
            template_ids=tids, n_instances_per_template=5
        )
        assert set(labels) == set(tids)
        assert len(queries) == 10


# ---------------------------------------------------------------------------
# ProvenanceRecord
# ---------------------------------------------------------------------------

class TestProvenanceRecord:
    def _make_record(self):
        return ProvenanceRecord(
            query_id="q001_0000",
            template_id=1,
            sql="SELECT 1",
            lineage_data={"orders": [1, 2], "customer": [5]},
            atomic_why_data=[[["orders", 1], ["customer", 5]]],
            blocked_why_data=[{"orders": [1, 2], "customer": [5]}],
        )

    def test_lineage_property(self):
        record = self._make_record()
        lin = record.lineage
        assert ("orders", 1) in lin
        assert ("customer", 5) in lin

    def test_atomic_why_property(self):
        record = self._make_record()
        wp = record.atomic_why_provenance
        assert len(wp) == 1

    def test_blocked_why_property(self):
        record = self._make_record()
        bwp = record.blocked_why_provenance
        assert len(bwp) == 1

    def test_round_trip_dict(self):
        record = self._make_record()
        d = record.to_dict()
        restored = ProvenanceRecord.from_dict(d)
        assert restored.query_id == record.query_id
        assert restored.template_id == record.template_id
        assert restored.lineage_data == record.lineage_data


# ---------------------------------------------------------------------------
# ProvenanceDataset
# ---------------------------------------------------------------------------

class TestProvenanceDataset:
    def _make_dataset(self):
        records = [
            ProvenanceRecord("q1", 1, "SQL1", {"t": [1]}, [], []),
            ProvenanceRecord("q2", 1, "SQL2", {"t": [2]}, [], []),
            ProvenanceRecord("q3", 2, "SQL3", {"u": [10]}, [], []),
        ]
        return ProvenanceDataset(records)

    def test_len(self):
        ds = self._make_dataset()
        assert len(ds) == 3

    def test_iter(self):
        ds = self._make_dataset()
        ids = [r.query_id for r in ds]
        assert ids == ["q1", "q2", "q3"]

    def test_getitem(self):
        ds = self._make_dataset()
        assert ds[0].query_id == "q1"

    def test_template_ids(self):
        ds = self._make_dataset()
        assert ds.template_ids == [1, 1, 2]

    def test_filter_by_template(self):
        ds = self._make_dataset()
        filtered = ds.filter_by_template(1)
        assert len(filtered) == 2
        assert all(r.template_id == 1 for r in filtered)

    def test_lineages_batch(self):
        ds = self._make_dataset()
        lins = ds.lineages
        assert len(lins) == 3


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

class TestJsonlIO:
    def _make_dataset(self):
        records = [
            ProvenanceRecord(
                "q1", 1, "SELECT 1",
                {"orders": [1, 2]},
                [[["orders", 1]]],
                [{"orders": [1, 2]}],
            ),
            ProvenanceRecord(
                "q2", 2, "SELECT 2",
                {"customer": [5]},
                [[["customer", 5]]],
                [{"customer": [5]}],
            ),
        ]
        return ProvenanceDataset(records)

    def test_save_and_load(self):
        ds = self._make_dataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prov.jsonl"
            save_provenance_jsonl(ds, path)
            loaded = load_provenance_jsonl(path)

        assert len(loaded) == len(ds)
        for orig, restored in zip(ds, loaded):
            assert orig.query_id == restored.query_id
            assert orig.template_id == restored.template_id
            assert orig.lineage_data == restored.lineage_data

    def test_file_is_valid_jsonl(self):
        ds = self._make_dataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prov.jsonl"
            save_provenance_jsonl(ds, path)
            lines = path.read_text().strip().splitlines()
            assert len(lines) == len(ds)
            for line in lines:
                obj = json.loads(line)
                assert "query_id" in obj
                assert "template_id" in obj

    def test_creates_parent_directories(self):
        ds = self._make_dataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "sub" / "dir" / "prov.jsonl"
            save_provenance_jsonl(ds, nested)
            assert nested.exists()
