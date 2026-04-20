"""MSSQL writer utilities for appending synthetic data to the target db."""

from kubota_synth.load.mssql_writer import MSSQLWriter, write_synthetic

__all__ = ["MSSQLWriter", "write_synthetic"]
