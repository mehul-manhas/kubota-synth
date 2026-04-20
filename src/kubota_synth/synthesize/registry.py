"""Registry of supported SDV synthesizers and the factory that builds them."""

from __future__ import annotations

import copy
import logging
from typing import Any

from sdv.metadata import SingleTableMetadata
from sdv.sequential import PARSynthesizer
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer

from kubota_synth.config import TableConfig

logger = logging.getLogger(__name__)


# Map friendly config names -> SDV synthesizer classes.
SYNTHESIZER_CLASSES: dict[str, type] = {
    "GaussianCopula": GaussianCopulaSynthesizer,
    "CTGAN": CTGANSynthesizer,
    "TVAE": TVAESynthesizer,
    "PAR": PARSynthesizer,
}


def _strip_internal_fields(metadata_dict: dict[str, Any]) -> tuple[dict[str, Any], list[Any]]:
    """Split a table metadata dict into (public_metadata, constraints)."""
    md = copy.deepcopy(metadata_dict)
    constraints = md.pop("_constraints", []) or []
    return md, constraints


def build_synthesizer(table_cfg: TableConfig, table_metadata: dict[str, Any]):
    """Instantiate an SDV synthesizer for ``table_cfg`` using ``table_metadata``."""
    name = table_cfg.synthesizer
    if name not in SYNTHESIZER_CLASSES:
        raise ValueError(
            f"Unknown synthesizer {name!r}. "
            f"Valid choices: {sorted(SYNTHESIZER_CLASSES)}"
        )

    md_dict, constraints = _strip_internal_fields(table_metadata)
    metadata = SingleTableMetadata.load_from_dict(md_dict)

    cls = SYNTHESIZER_CLASSES[name]
    kwargs: dict[str, Any] = {}

    if name in {"CTGAN", "TVAE"}:
        if table_cfg.epochs is not None:
            kwargs["epochs"] = int(table_cfg.epochs)

    if name == "PAR":
        if not table_cfg.sequence_key:
            raise ValueError(
                f"Table '{table_cfg.name}' is configured to use the PAR synthesizer "
                "but no `sequence_key` was provided. Add a `sequence_key` (and "
                "optionally a `sequence_index`) to the table config."
            )
        metadata.set_sequence_key(table_cfg.sequence_key)
        if table_cfg.sequence_index:
            metadata.set_sequence_index(table_cfg.sequence_index)
        if table_cfg.epochs is not None:
            kwargs["epochs"] = int(table_cfg.epochs)

    synth = cls(metadata=metadata, **kwargs)

    if constraints:
        try:
            synth.add_constraints(constraints=constraints)
            logger.info(
                "Applied %d constraint(s) to %s synthesizer.", len(constraints), name
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to apply constraints for table %s: %s", table_cfg.name, exc
            )

    return synth
