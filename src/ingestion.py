"""Input loading and light structural validation for OpsReport."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, IO, Any

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xlsm"}


class DataIngestionError(ValueError):
    """Raised when an uploaded dataset cannot be loaded safely."""


def load_data(
    source: str | Path | bytes | bytearray | BinaryIO | IO[bytes] | pd.DataFrame,
    filename: str | None = None,
    sheet_name: str | int = 0,
) -> pd.DataFrame:
    """Load a CSV/TSV/XLSX source into a validated dataframe.

    ``source`` accepts paths, raw bytes and Streamlit-style uploaded files. When
    bytes are supplied without ``filename``, Excel is detected by its ZIP magic
    signature and all other content is treated as delimited text.
    """
    if isinstance(source, pd.DataFrame):
        return validate_dataframe(source.copy())

    raw, inferred_name = _read_source(source)
    resolved_name = filename or inferred_name
    extension = Path(resolved_name).suffix.lower() if resolved_name else _detect_extension(raw)

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DataIngestionError(
            f"Formato no compatible: '{extension or 'desconocido'}'. Formatos admitidos: {supported}."
        )
    if not raw:
        raise DataIngestionError("El archivo está vacío.")

    try:
        if extension in {".xlsx", ".xlsm"}:
            dataframe = pd.read_excel(BytesIO(raw), sheet_name=sheet_name, engine="openpyxl")
        else:
            dataframe = _read_delimited(raw, extension)
    except DataIngestionError:
        raise
    except Exception as exc:  # pandas/openpyxl expose many parser-specific errors
        raise DataIngestionError(f"No fue posible leer el archivo: {exc}") from exc

    if isinstance(dataframe, dict):
        raise DataIngestionError("Se debe seleccionar una sola hoja de Excel.")
    return validate_dataframe(dataframe)


def validate_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply non-destructive structural checks and normalize header whitespace."""
    if dataframe.empty and len(dataframe.columns) == 0:
        raise DataIngestionError("El archivo no contiene columnas ni registros.")

    result = dataframe.copy()
    normalized_columns: list[str] = []
    for index, column in enumerate(result.columns, start=1):
        label = str(column).replace("\ufeff", "").strip()
        normalized_columns.append(label or f"column_{index}")

    duplicates = _duplicates(normalized_columns)
    if duplicates:
        duplicate_text = ", ".join(duplicates)
        raise DataIngestionError(f"Hay nombres de columnas duplicados: {duplicate_text}.")

    result.columns = normalized_columns
    if result.empty:
        raise DataIngestionError("El archivo contiene columnas, pero no tiene registros.")
    return result


def _read_source(source: Any) -> tuple[bytes, str | None]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            return path.read_bytes(), path.name
        except OSError as exc:
            raise DataIngestionError(f"No fue posible abrir '{path}': {exc}") from exc

    if isinstance(source, (bytes, bytearray)):
        return bytes(source), None

    inferred_name = getattr(source, "name", None)
    if hasattr(source, "getvalue"):
        value = source.getvalue()
        if isinstance(value, str):
            value = value.encode("utf-8")
        return bytes(value), inferred_name

    if hasattr(source, "read"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            value = source.read()
            if hasattr(source, "seek"):
                source.seek(0)
        except (OSError, ValueError) as exc:
            raise DataIngestionError(f"No fue posible leer el archivo: {exc}") from exc
        if isinstance(value, str):
            value = value.encode("utf-8")
        return bytes(value), inferred_name

    raise DataIngestionError("La fuente de datos entregada no es compatible.")


def _read_delimited(raw: bytes, extension: str) -> pd.DataFrame:
    encodings = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
    last_error: Exception | None = None
    separator = "\t" if extension == ".tsv" else None

    for encoding in encodings:
        try:
            return pd.read_csv(
                BytesIO(raw),
                encoding=encoding,
                sep=separator,
                engine="python" if separator is None else "c",
            )
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.ParserError as exc:
            last_error = exc

    raise DataIngestionError(f"No fue posible interpretar el archivo de texto: {last_error}")


def _detect_extension(raw: bytes) -> str:
    return ".xlsx" if raw[:4] == b"PK\x03\x04" else ".csv"


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates
