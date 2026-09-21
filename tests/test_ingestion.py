from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest

from src.ingestion import DataIngestionError, load_data


def test_load_csv_from_bytes_detects_delimiter_and_trims_headers() -> None:
    raw = " operation_id ;region;revenue_clp\nOP-1;Centro;125000\nOP-2;Sur;98000\n".encode()

    result = load_data(raw, filename="operations.csv")

    assert result.columns.tolist() == ["operation_id", "region", "revenue_clp"]
    assert result.shape == (2, 3)
    assert result.loc[0, "revenue_clp"] == 125000


def test_load_excel_from_streamlit_like_file() -> None:
    expected = pd.DataFrame({"operation_id": ["OP-1", "OP-2"], "status": ["Completada", "Cancelada"]})
    buffer = BytesIO()
    expected.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)
    buffer.name = "operations.xlsx"

    result = load_data(buffer)

    pd.testing.assert_frame_equal(result, expected)


def test_load_excel_bytes_without_filename_uses_magic_signature() -> None:
    expected = pd.DataFrame({"value": [1, 2, 3]})
    buffer = BytesIO()
    expected.to_excel(buffer, index=False, engine="openpyxl")

    result = load_data(buffer.getvalue())

    pd.testing.assert_frame_equal(result, expected)


def test_rejects_unsupported_extension() -> None:
    with pytest.raises(DataIngestionError, match="Formato no compatible"):
        load_data(b"a,b\n1,2\n", filename="operations.json")


def test_rejects_empty_dataset() -> None:
    with pytest.raises(DataIngestionError, match="vacío"):
        load_data(b"", filename="operations.csv")


def test_rejects_malformed_csv_with_unterminated_quote() -> None:
    raw = b'order_id,region\n"OP-1,Centro\n'

    with pytest.raises(DataIngestionError, match="No fue posible"):
        load_data(raw, filename="operations.csv")


def test_rejects_headers_that_collide_after_trimming() -> None:
    raw = b"region, region \nCentro,Sur\n"

    with pytest.raises(DataIngestionError, match="duplicados"):
        load_data(raw, filename="operations.csv")
