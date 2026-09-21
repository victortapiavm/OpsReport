# OpsReport Roadmap

This document tracks the planned evolution of OpsReport after the initial MVP.

The roadmap is intentionally flexible. A phase may be adjusted, split, merged, or reordered when implementation evidence shows that a different sequence would produce a better product. Completed work should remain documented so future development sessions can recover context quickly.

## Project principle

OpsReport should remain a small, credible operational-analysis product:

> Turn a messy business spreadsheet into a useful operational report in seconds.

The analytical engine remains deterministic and testable. Optional AI functionality may interpret or explain computed results, but it must not become the source of truth for numerical analysis.

---

## Phase 1 — Strong MVP

**Status: COMPLETE**

### Objective

Build the first complete, portfolio-quality version of OpsReport.

### Delivered

- Spanish Streamlit interface.
- CSV and XLSX ingestion.
- Bundled deterministic synthetic operational dataset.
- Data preview and detected date range.
- Missing-value and duplicate diagnostics.
- Practical deterministic 0–100 data-quality score.
- Column profiling and type inspection.
- KPI calculations outside the UI layer: revenue, orders, average order value, units, gross margin, cancellation rate, and processing time.
- Interactive Plotly visualizations.
- Deterministic anomaly detection.
- Deterministic Spanish executive summary.
- Excel export.
- Graceful handling of malformed or incomplete inputs.
- Automated tests.
- Bilingual, recruiter-friendly README.

### Validation baseline

At Phase 1 completion:

- 23 pytest tests passed.
- The bundled sample completed the full analysis/export path.
- Streamlit application smoke testing completed without UI exceptions.
- A real Streamlit server startup returned HTTP 200.
- Python compile checks passed.
- `git diff --check` passed.

This validation baseline should be preserved in later phases.

---

## Phase 2 — Portfolio Polish and Deployment

**Status: IN PROGRESS — GitHub publication and CI complete; public Streamlit deployment pending**

### Objective

Turn the working MVP into something that can be confidently linked from a CV, LinkedIn profile, GitHub profile, or job application.

### Planned work

- Review the complete UI visually and correct obvious spacing, typography, chart, and responsive-layout issues.
- Capture real screenshots of the running application.
- Add the best screenshots to the README.
- Improve the README opening section for technical recruiters.
- Add a concise architecture diagram if it improves comprehension.
- Verify installation instructions from a clean environment.
- Create the initial Git commit history cleanly.
- Publish the repository to GitHub.
- Add repository description, topics, and project URL.
- Deploy a live public demo using a simple Streamlit-compatible host.
- Verify the deployed sample-data workflow and Excel export.
- Add the live-demo link to the README.

### Definition of Done

Phase 2 is complete when:

- the repository is publicly presentable;
- the README contains real screenshots;
- installation instructions are verified;
- the application is available through a public demo URL;
- the public demo can analyze the bundled sample successfully;
- the repository and demo links are suitable for a professional profile.

### Scope guardrail

Do not add major analytical features merely to make the project look larger. This phase is about presentation, accessibility, reproducibility, and credibility.

---

## Phase 3 — Arbitrary Spreadsheet Support

**Status: COMPLETE**

### Objective

Make OpsReport useful for operational spreadsheets that do not already use the bundled sample's column names.

### Planned work

- Add a schema-recognition layer that proposes likely semantic roles for columns.
- Add an interactive column-mapping step when automatic inference is uncertain.
- Support mapping of order ID, date, revenue, cost, units, status, region, category, and processing time.
- Show which mappings were inferred automatically and which require confirmation.
- Allow partial mappings.
- Recalculate available KPIs from the resolved mapping.
- Keep unavailable KPIs explicit instead of failing.
- Improve type inconsistency diagnostics for numeric and date fields.
- Add deterministic tests for English headers, Spanish headers, ambiguous columns, missing semantic fields, and user-overridden mappings.

### Definition of Done

Phase 3 is complete when a spreadsheet with substantially different column names can be mapped through the UI and analyzed without editing source code.

### Delivered

- Added `src/schema.py` as the shared semantic authority for order ID, date, revenue, cost, units, status, region, category, and processing time.
- Added explainable Spanish/English header recognition with confidence, alternatives, and ambiguity handling.
- High-confidence mappings apply automatically; uncertain suggestions require confirmation.
- Added an interactive Streamlit mapping form with manual overrides, explicit unmapping, duplicate-source validation, and partial mappings.
- Threaded the resolved mapping through quality checks, KPIs, date range, charts, anomaly detection, executive narrative, and Excel export.
- Added mapped numeric and date inconsistency diagnostics without changing the documented quality-score weights.
- Unified cancellation interpretation across KPI, visualization, and narrative calculations.
- Preserved raw source headers in preview and export while carrying the semantic mapping as analysis context.
- Added deterministic tests for English headers, Spanish headers, ambiguity, missing roles, overrides, explicit unmapping, normalized-name collisions, type diagnostics, and full mapped analysis/export.
- Added a Streamlit AppTest proving a CSV with arbitrary headers can be uploaded, mapped through the UI, analyzed, and exported without source-code changes.

### Validation at completion

- 31 pytest tests passed in the project virtual environment.
- The arbitrary-schema Streamlit AppTest completed without UI exceptions.
- The bundled sample still auto-maps all 9 supported semantic roles and renders the existing dashboard flow.
- Python compile checks passed during implementation.

### Scope guardrail

Column recognition should remain explainable. Avoid introducing a mandatory LLM merely to guess column names.

---

## Phase 4 — Reporting and Analytical Depth

**Status: PLANNED**

### Objective

Increase the usefulness of the report for an operations analyst or junior management user while keeping the analytical methods understandable.

### Candidate work

- Period-over-period comparison with absolute and percentage change.
- Clear handling of incomplete periods.
- Segment comparisons across region, category, channel, or status.
- Cancellation-rate diagnostics by segment.
- Margin analysis by segment.
- Processing-time SLA diagnostics where an SLA can be mapped or configured.
- Better time-series anomaly detection using rolling or historical baselines.
- Configurable deterministic thresholds.
- Additional export tables reflecting the new analysis.
- Optional lightweight PDF export if it can be implemented reliably and portably.

### Definition of Done

Phase 4 is complete when OpsReport can answer basic comparative questions such as:

- What changed versus the previous period?
- Which segment deteriorated the most?
- Which segment has the highest cancellation rate?
- Where are processing times unusually high?
- Which segment contributes the most revenue or margin?

The answers must remain traceable to deterministic calculations.

### Scope guardrail

Do not turn OpsReport into a general-purpose BI platform. Add analyses that directly support the spreadsheet-to-report workflow.

---

## Phase 5 — Optional AI Interpretation Layer

**Status: PLANNED / OPTIONAL**

### Objective

Add a natural-language analyst layer that explains and contextualizes results already calculated by OpsReport.

The deterministic engine remains the source of truth.

### Architecture

    CSV / XLSX
        ↓
    Ingestion and validation
        ↓
    Deterministic profiling, KPIs and anomaly detection
        ↓
    Structured analysis facts
        ↓
    Deterministic executive narrative
        ↓
    Optional LLM interpretation layer

The LLM receives structured facts. It does not independently calculate revenue, margin, cancellation rates, outliers, or other business metrics.

### Example structured input

    Total revenue: 125,000,000 CLP
    Gross margin rate: 28.4%
    Cancellation rate: 9.7%
    Average processing time: 18.6 h

    Findings:
    - La Araucanía cancellation rate: 19.2%
    - Repuestos has the highest average processing time
    - A processing-time observation is 2.3x the period average

An optional LLM can turn those facts into a clearer management narrative, answer questions about them, or suggest areas for investigation.

### Planned capabilities

- Optional enhanced executive summary.
- "Ask OpsReport" analysis chat grounded in computed facts.
- Plain-language explanations of anomaly findings.
- Questions such as:
  - "¿Qué debería investigar primero?"
  - "¿Qué región presenta más señales de deterioro?"
  - "¿Por qué OpsReport marcó este valor como anómalo?"
  - "Resume esto para una reunión de gerencia."
- Narrative modes such as executive, operational, and technical.
- Explicit provenance from generated statements back to calculated metrics or findings where practical.

### Analytical constraints

The AI layer must:

- never silently replace deterministic calculations;
- never invent unavailable values;
- avoid unsupported causal claims;
- distinguish observations from hypotheses;
- remain optional so OpsReport still works without an API key or internet connection;
- fall back to the deterministic narrative when AI is unavailable.

Acceptable:

> La Araucanía presenta una tasa de cancelación superior al promedio. Conviene investigar variables operativas asociadas.

Unsupported by the spreadsheet alone:

> La Araucanía tiene más cancelaciones debido a problemas logísticos.

### Definition of Done

Phase 5 is complete when:

- the deterministic application still works without an LLM;
- the AI receives only structured analytical context needed for the request;
- generated explanations do not alter source metrics;
- unsupported causal language is constrained;
- failures or missing API configuration fall back cleanly;
- representative behavior is covered by tests around context construction and fallback behavior.

---

## Possible Later Work

These are ideas rather than committed phases:

- reusable analysis presets for different operational domains;
- saved configuration files for recurring spreadsheet formats;
- side-by-side comparison of multiple files;
- richer export templates;
- CI/CD for tests and deployment;
- optional local-model support for the AI interpretation layer.

Any later feature should earn its complexity by improving the core spreadsheet-to-report workflow or the project's value as a software-engineering portfolio piece.

---

## Current Next Step

Finish the remaining **Phase 2 — Portfolio Polish and Deployment** item: complete Streamlit Community Cloud authorization/deployment, verify the bundled sample and Excel export on the public instance, then add the live-demo URL to GitHub and this README.

After that, **Phase 4 — Reporting and Analytical Depth** is the next product-development phase unless implementation evidence suggests a better intermediate step.
