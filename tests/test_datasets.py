import pytest

from src.datasets import DatasetItem, DatasetTier, EvalDataset


class TestDatasetItem:
    def test_golden_item_requires_verification(self):
        with pytest.raises(ValueError):
            DatasetItem("X1", DatasetTier.GOLDEN, "q", "a", verified_by_human=False)

    def test_golden_item_with_verification_is_fine(self):
        item = DatasetItem("X1", DatasetTier.GOLDEN, "q", "a", verified_by_human=True)
        assert item.tier == DatasetTier.GOLDEN

    def test_silver_item_does_not_require_verification(self):
        item = DatasetItem("X1", DatasetTier.SILVER, "q", "a", verified_by_human=False)
        assert item.tier == DatasetTier.SILVER
        assert item.verified_by_human is False

    def test_silver_item_can_also_be_verified(self):
        item = DatasetItem("X1", DatasetTier.SILVER, "q", "a", verified_by_human=True)
        assert item.verified_by_human is True


class TestEvalDataset:
    def test_add_and_get(self):
        ds = EvalDataset("test")
        item = DatasetItem("A1", DatasetTier.GOLDEN, "q", "a", verified_by_human=True)
        ds.add(item)
        assert ds.get("A1") is item

    def test_duplicate_id_raises(self):
        ds = EvalDataset("test")
        ds.add(DatasetItem("A1", DatasetTier.GOLDEN, "q", "a", verified_by_human=True))
        with pytest.raises(ValueError):
            ds.add(DatasetItem("A1", DatasetTier.SILVER, "q2", "a2", verified_by_human=False))

    def test_get_unknown_id_raises(self):
        ds = EvalDataset("test")
        with pytest.raises(KeyError):
            ds.get("NOT_REAL")

    def test_golden_items_and_silver_items_partition_correctly(self):
        ds = EvalDataset("test")
        ds.add(DatasetItem("G1", DatasetTier.GOLDEN, "q1", "a1", verified_by_human=True))
        ds.add(DatasetItem("G2", DatasetTier.GOLDEN, "q2", "a2", verified_by_human=True))
        ds.add(DatasetItem("S1", DatasetTier.SILVER, "q3", "a3", verified_by_human=False))

        golden_ids = {i.item_id for i in ds.golden_items()}
        silver_ids = {i.item_id for i in ds.silver_items()}
        assert golden_ids == {"G1", "G2"}
        assert silver_ids == {"S1"}

    def test_len_reflects_total_items(self):
        ds = EvalDataset("test")
        ds.add(DatasetItem("G1", DatasetTier.GOLDEN, "q", "a", verified_by_human=True))
        ds.add(DatasetItem("S1", DatasetTier.SILVER, "q", "a", verified_by_human=False))
        assert len(ds) == 2


class TestPromoteToGolden:
    def test_promotes_a_verified_silver_item(self):
        ds = EvalDataset("test")
        ds.add(DatasetItem("S1", DatasetTier.SILVER, "q", "a", verified_by_human=True))
        promoted = ds.promote_to_golden("S1")
        assert promoted.tier == DatasetTier.GOLDEN
        assert ds.get("S1").tier == DatasetTier.GOLDEN

    def test_refuses_to_promote_unverified_silver_item(self):
        ds = EvalDataset("test")
        ds.add(DatasetItem("S1", DatasetTier.SILVER, "q", "a", verified_by_human=False))
        with pytest.raises(ValueError):
            ds.promote_to_golden("S1")

    def test_refuses_to_promote_already_golden_item(self):
        ds = EvalDataset("test")
        ds.add(DatasetItem("G1", DatasetTier.GOLDEN, "q", "a", verified_by_human=True))
        with pytest.raises(ValueError):
            ds.promote_to_golden("G1")

    def test_promoted_item_preserves_content(self):
        ds = EvalDataset("test")
        ds.add(DatasetItem("S1", DatasetTier.SILVER, "question text", "answer text", verified_by_human=True, tags=["t1"]))
        promoted = ds.promote_to_golden("S1")
        assert promoted.input_text == "question text"
        assert promoted.expected_output == "answer text"
        assert promoted.tags == ["t1"]
