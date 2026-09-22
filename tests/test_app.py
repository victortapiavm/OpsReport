from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).parents[1] / "app.py"


def test_bundled_sample_renders_phase4_analysis() -> None:
    app = AppTest.from_file(APP_PATH).run(timeout=20)
    next(button for button in app.button if button.label == "Probar con datos de ejemplo").click()
    app.run(timeout=30)

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["SLA configurado"] == "24,0 h"
    assert "Tasa de incumplimiento" in metrics
    assert any(selectbox.label == "Dimensión de segmento" for selectbox in app.selectbox)
    assert app.get("download_button")


def test_arbitrary_upload_can_be_mapped_through_ui() -> None:
    csv_bytes = (
        "Ref,When,Billings,Spend,Pieces,Lifecycle,Market,Line,Cycle\n"
        "A,2026-09-01,100,60,1,Done,N,A,2\n"
        "B,2026-09-02,102,61,2,Done,N,A,2\n"
        "C,2026-09-03,98,58,1,Cancelled,S,B,2\n"
        "D,2026-09-04,101,60,1,Done,S,B,2\n"
        "E,2026-09-05,1000,400,3,Done,S,B,20\n"
    ).encode()

    app = AppTest.from_file(APP_PATH).run(timeout=20)
    app.file_uploader[0].upload("arbitrary_operations.csv", csv_bytes, "text/csv")
    app.run(timeout=20)
    assert not app.exception

    selections = {
        "ID de pedido": "Ref",
        "Fecha": "When",
        "Ingresos": "Billings",
        "Costo": "Spend",
        "Unidades": "Pieces",
        "Estado": "Lifecycle",
        "Región": "Market",
        "Categoría": "Line",
        "Tiempo de procesamiento (h)": "Cycle",
    }
    for label, source_column in selections.items():
        next(item for item in app.selectbox if item.label == label).select(source_column)

    next(button for button in app.button if button.label == "Aplicar mapeo").click()
    app.run(timeout=30)

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Ingresos totales"] == "$1.401"
    assert metrics["Pedidos"] == "5"
    assert metrics["Cancelación"] == "20,0%"
    assert metrics["Período"] == "01-09-2026 → 05-09-2026"
    assert app.get("download_button")
