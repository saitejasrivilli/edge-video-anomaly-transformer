"""Dataset ingestion and validation layer.

This package separates dataset concerns into distinct modules:

- ``paths``: environment-agnostic (local vs. Colab) path configuration
- ``schemas``: typed manifest record definitions
- ``manifests``: manifest generation and JSONL read/write
- ``validation``: structural and referential validation of manifests
- ``datasets``: dataset-specific adapters (one module per dataset)
"""
