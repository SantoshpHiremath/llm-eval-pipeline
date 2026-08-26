"""
Golden and Silver dataset management -- a real, two-tier distinction (not
present in any prior project in this portfolio), matching how evaluation
datasets are actually used in practice:

- GOLDEN examples are hand-verified, high-confidence ground truth: a human
  has reviewed the input, the expected output, and signed off that the
  expected output is genuinely correct. These are the examples an eval
  pipeline trusts most, and regressions against them should block a
  release.
- SILVER examples are lower-confidence: often auto-collected from real
  traffic (a production input, paired with what a currently-trusted model
  version produced), useful for catching regressions and covering more
  input diversity than a small hand-curated golden set ever could, but
  NOT to be treated as ground truth with the same confidence -- a silver
  "expected" answer could itself be wrong, so silver-set failures should
  be reviewed, not automatically treated as bugs.

This distinction is a real, common practice in LLM evaluation (echoed in
tools like Langfuse's dataset items, which support exactly this kind of
tiered trust), not an invented category.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class DatasetTier(str, Enum):
    GOLDEN = "golden"
    SILVER = "silver"


@dataclass
class DatasetItem:
    item_id: str
    tier: DatasetTier
    input_text: str
    expected_output: str
    # Golden items are always verified; silver items may or may not be
    # (e.g. reviewed after being auto-collected, or still pending review).
    verified_by_human: bool
    tags: list = field(default_factory=list)
    created_at: str = ""

    def __post_init__(self):
        if self.tier == DatasetTier.GOLDEN and not self.verified_by_human:
            raise ValueError(
                f"Golden item {self.item_id} must be human-verified — "
                "an unverified 'golden' item is a contradiction in terms."
            )


class EvalDataset:
    """A named collection of golden + silver items, with helpers to filter
    by tier and to promote a silver item to golden once a human has
    reviewed and confirmed it -- the real workflow by which a silver set
    grows the golden set over time, rather than the two tiers being
    static forever.
    """

    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, DatasetItem] = {}

    def add(self, item: DatasetItem) -> None:
        if item.item_id in self._items:
            raise ValueError(f"Duplicate item_id: {item.item_id}")
        self._items[item.item_id] = item

    def get(self, item_id: str) -> DatasetItem:
        if item_id not in self._items:
            raise KeyError(f"Unknown item_id: {item_id}")
        return self._items[item_id]

    def all_items(self) -> list:
        return list(self._items.values())

    def golden_items(self) -> list:
        return [i for i in self._items.values() if i.tier == DatasetTier.GOLDEN]

    def silver_items(self) -> list:
        return [i for i in self._items.values() if i.tier == DatasetTier.SILVER]

    def promote_to_golden(self, item_id: str) -> DatasetItem:
        """Promotes a silver item to golden -- requires it to already be
        human-verified (promotion itself is not the act of verification;
        a human must have reviewed it first, then promotion just changes
        its trust tier).
        """
        item = self.get(item_id)
        if item.tier == DatasetTier.GOLDEN:
            raise ValueError(f"{item_id} is already golden")
        if not item.verified_by_human:
            raise ValueError(
                f"Cannot promote {item_id} to golden: not yet human-verified. "
                "Review it and set verified_by_human=True first."
            )
        promoted = DatasetItem(
            item_id=item.item_id,
            tier=DatasetTier.GOLDEN,
            input_text=item.input_text,
            expected_output=item.expected_output,
            verified_by_human=True,
            tags=item.tags,
            created_at=item.created_at,
        )
        self._items[item_id] = promoted
        return promoted

    def __len__(self) -> int:
        return len(self._items)
