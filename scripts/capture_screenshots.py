from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright

from yakupredict_ai.schemas import Measurement
from yakupredict_ai.service import evaluate_measurement, get_model
from yakupredict_ai.storage import clear_predictions, recent_predictions

OUTPUT = ROOT / "screenshots"
OUTPUT.mkdir(parents=True, exist_ok=True)

PRESETS = {
    "normal": {
        "flow_m3s": 78, "power_mw": 72, "vibration_mms": 2.2,
        "bearing_temp_c": 66, "oil_temp_c": 52, "hydraulic_pressure_bar": 74,
        "wicket_gate_pct": 75, "stator_current_a": 1360, "ambient_temp_c": 22,
    },
    "watch": {
        "flow_m3s": 76, "power_mw": 69, "vibration_mms": 4.1,
        "bearing_temp_c": 76, "oil_temp_c": 59, "hydraulic_pressure_bar": 70,
        "wicket_gate_pct": 73, "stator_current_a": 1420, "ambient_temp_c": 23,
    },
    "critical": {
        "flow_m3s": 80, "power_mw": 67, "vibration_mms": 6.4,
        "bearing_temp_c": 88, "oil_temp_c": 67, "hydraulic_pressure_bar": 65,
        "wicket_gate_pct": 78, "stator_current_a": 1510, "ambient_temp_c": 23,
    },
}


def _standalone_html() -> str:
    html = (ROOT / "templates" / "dashboard.html").read_text(encoding="utf-8")
    css = (ROOT / "static" / "app.css").read_text(encoding="utf-8")
    html = html.replace('{{ software_name }}', 'YakuPredict AI').replace('{{ version }}', '3.0.0')
    html = html.replace('<link rel="stylesheet" href="/static/app.css" />', f'<style>{css}</style>')
    html = html.replace('<script src="/static/app.js"></script>', '')
    return html


def _fill(page, result, values, history):
    for key, value in values.items():
        page.locator(f'input[name="{key}"]').fill(str(value))

    page.locator('#risk-score').evaluate("(el, v) => el.textContent = v", f"{result.risk_percent:.2f} %")
    page.locator('#risk-bar').evaluate("(el, v) => el.style.width = v", f"{result.risk_percent}%")
    page.locator('#risk-level').evaluate("(el, v) => {el.textContent=v.text; el.className='level-badge '+v.cls}", {
        "text": result.level,
        "cls": {"BAJO":"low","MEDIO":"medium","ALTO":"high","CRÍTICO":"critical"}[result.level],
    })
    page.locator('#supervised-score').evaluate("(el, v) => el.textContent = v", f"{result.supervised_probability*100:.1f} %")
    page.locator('#anomaly-score').evaluate("(el, v) => el.textContent = v", f"{result.anomaly_probability*100:.1f} %")
    page.locator('#decision-caption').evaluate("(el, v) => el.textContent = v", f"Registro #{result.id}")
    page.locator('#recommendation-state').evaluate("el => el.classList.add('hidden')")
    page.locator('#recommendation-result').evaluate("el => el.classList.remove('hidden')")
    page.locator('#decision-level').evaluate("(el, v) => el.textContent = v", result.level)
    page.locator('#recommendation-text').evaluate("(el, v) => el.textContent = v", result.recommendation)
    page.locator('#rf-breakdown').evaluate("(el, v) => el.textContent = v", f"{result.supervised_probability*100:.1f} %")
    page.locator('#if-breakdown').evaluate("(el, v) => el.textContent = v", f"{result.anomaly_probability*100:.1f} %")

    rows = []
    for item in history:
        rows.append(
            f"<tr><td>#{item['id']}</td><td>{item['created_at'][0:19].replace('T',' ')}</td>"
            f"<td class='risk-chip'>{item['risk_percent']:.2f} %</td>"
            f"<td><span class='level-chip {item['level'].lower()}'>{item['level']}</span></td>"
            f"<td>{item['supervised_probability']*100:.1f} %</td>"
            f"<td>{item['anomaly_probability']*100:.1f} %</td></tr>"
        )
    page.locator('#history-body').evaluate("(el, html) => el.innerHTML = html", ''.join(rows))


def capture() -> None:
    clear_predictions()
    get_model.cache_clear()
    nominal = evaluate_measurement(Measurement(**PRESETS["normal"]))
    watch = evaluate_measurement(Measurement(**PRESETS["watch"]))
    critical = evaluate_measurement(Measurement(**PRESETS["critical"]))
    history = recent_predictions(8)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path="/usr/bin/chromium", args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 980}, device_scale_factor=1)

        page.set_content(_standalone_html(), wait_until="load")
        _fill(page, nominal, PRESETS["normal"], history)
        page.screenshot(path=str(OUTPUT / "01_dashboard_condicion_normal.png"), full_page=True)

        page.set_content(_standalone_html(), wait_until="load")
        _fill(page, critical, PRESETS["critical"], history)
        page.screenshot(path=str(OUTPUT / "02_dashboard_condicion_critica.png"), full_page=True)

        page.set_content(_standalone_html(), wait_until="load")
        _fill(page, watch, PRESETS["watch"], history)
        page.locator('#history-panel').scroll_into_view_if_needed()
        page.screenshot(path=str(OUTPUT / "03_dashboard_historial.png"), full_page=True)
        browser.close()

    print("Capturas generadas con el modelo entrenado:")
    print(f"  Normal: {nominal.risk_percent:.2f}% ({nominal.level})")
    print(f"  Seguimiento: {watch.risk_percent:.2f}% ({watch.level})")
    print(f"  Crítico: {critical.risk_percent:.2f}% ({critical.level})")


if __name__ == "__main__":
    capture()
