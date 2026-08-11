from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Protocol

from cosmit.domain.models import ComponentId, TestKnowledge


class FrameworkAdapter(Protocol):
    """Framework-specific boundary used by the six-stage pipeline."""

    @property
    def framework_id(self) -> str:
        """Return the stable framework identifier."""

    def scan_components(self, repository: Path) -> Iterable[ComponentId]:
        """Discover public and internal components relevant to testing."""

    def collect_component_metadata(self, component: ComponentId) -> Mapping[str, object]:
        """Collect documentation, signature, source, usage, history and test evidence."""

    def extract_test_knowledge(self, component: ComponentId) -> Iterable[TestKnowledge]:
        """Extract structured test knowledge associated with a component."""

    def render_migrated_test(
        self,
        knowledge: TestKnowledge,
        target_component: ComponentId,
        transformations: Mapping[str, object],
    ) -> str:
        """Render a target-framework-native test while preserving traceability."""
