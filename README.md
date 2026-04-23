# Clinical Decision Support Dashboard

A rule-based care gap detection system and interactive dashboard built on **MIMIC-IV**, with FHIR export support for hospital interoperability.

---

## Overview

This project ingests raw MIMIC-IV clinical data, identifies potential care gaps using a rule-based engine, and presents findings through an interactive Streamlit dashboard designed for clinicians. Patient data is also exported as FHIR Transaction Bundles for integration with standard healthcare systems (e.g., HAPI FHIR, Epic).

---

## Data Pipeline

```
MIMIC-IV (raw CSVs)
       │
       ▼
build_profiles.py          → admission_profile.csv (546K admissions)
                             patient_summary.csv   (364K patients)
                             patient_dict.jsonl    (223K patients)
       │
       ▼
build_subset.py            → processed/subset/    (1,000 patients)
       │
       ├──▶ build_fhir.py  → fhir_output/         (1,000 FHIR bundles)
       │
       └──▶ app.py         → Streamlit dashboard
```

### Subset Selection Strategy
1,000 patients are selected across four non-overlapping groups to ensure clinical diversity:

| Group | Criteria | Count |
|---|---|---|
| Rich multi-admission | ≥2 admissions, ≥2 with 20+ labs | 400 |
| ICU experience | Has ICU stay, ≥10 labs | 300 |
| In-hospital death | ≥1 hospital death | 150 |
| Outpatient history | ≥10 OMR records, ≥2 admissions | 150 |

---

## Dashboard (`app.py`)

Run with:
```bash
streamlit run app.py
```

### Patient Search (Sidebar)
- Search by **Subject ID** or select from dropdown
- 1,000 patients loaded via byte-offset index (instant lookup without loading full file)

---

### Admission Timeline
A horizontal dot timeline showing all admissions for a patient on a single axis.

- Each dot represents one admission, positioned at the midpoint between admit and discharge date
- **Color** encodes admission type:
  - Red — Emergency (EW EMER., DIRECT EMER.)
  - Orange — Urgent
  - Purple — Observation
  - Blue — Surgical same day
- **Dot size** — selected admission is enlarged with a dark border for easy identification
- **Hover** shows: admission number, primary diagnosis, type, length of stay, ICU status
- No text labels on dots to avoid overlap in dense timelines

---

### Top Metrics Bar
Six key figures for the selected admission at a glance:

| Metric | Source field |
|---|---|
| LOS (days) | `los_hours / 24` |
| Diagnoses | `n_diagnoses` |
| Lab tests | `n_labs` |
| Abnormal labs | `n_abnormal_labs` |
| Medications | `n_prescriptions` |
| Care gaps | Output of rule engine |

---

### Tab 1 — Care Gaps
Displays all rule-triggered care gaps, sorted by severity (HIGH → MEDIUM → LOW).

Each gap is rendered as a color-coded card:
- **Red** border — HIGH severity
- **Orange** border — MEDIUM severity
- **Green** border — LOW severity

A bar chart summarizes gap counts by severity level.

**12 Rules in the engine:**

| Rule | Condition | Severity |
|---|---|---|
| R001 | Diabetes + no HbA1c | HIGH |
| R002 | Diabetes + no glucose | MEDIUM |
| R003 | Renal failure + no creatinine | HIGH |
| R004 | Liver disease + no ALT/AST | HIGH |
| R005 | HIV + no CD4 count | MEDIUM |
| R006 | Heart failure + no BNP | MEDIUM |
| R007 | Metformin + no creatinine | HIGH |
| R008 | Anticoagulant + no PT/INR | HIGH |
| R009 | BP > 140/90 on 2+ outpatient visits | MEDIUM |
| R010 | BMI < 18.5 on outpatient record | MEDIUM |
| R011 | ≥5% weight loss trend | LOW |
| R012 | >50% abnormal labs (min 10 tests) | HIGH |

---

### Tab 2 — Labs
**Left panel — Donut chart**
- Shows the ratio of normal vs. abnormal lab results
- Center label shows total test count
- Color: green = normal, red = abnormal
- Status message changes by abnormality rate: >50% → error, >30% → warning, else → success

**Right panel — Test list**
- All lab tests performed during admission, displayed in two columns

---

### Tab 3 — Diagnoses
- Primary diagnosis highlighted at the top with ICD code
- Full diagnosis list numbered, with ICD-9 or ICD-10 codes in a right column

---

### Tab 4 — Pre-admission Vitals
Data sourced from the **OMR (Outpatient Medical Records)** table, linked to each admission by calculating days elapsed before the admission date.

- **Table** — raw records (days before admission, measure name, value)
- **Trend chart** — line chart of numeric vitals over time (x-axis: days before admission, reversed so recent is on the right)
  - Blood Pressure → systolic value extracted
  - BMI, Weight, Height, eGFR plotted as-is
  - Up to 5 measures shown

---

### Tab 5 — Medications
- Full prescription list for the admission in two columns
- Procedure list below (if any procedures recorded)

---

### Tab 6 — AI Summary
- Displays the structured prompt that will be passed to an LLM
- LLM integration is model-agnostic — plug in any open-source model by implementing `get_llm_summary()`

Prompt includes: patient demographics, admission type, LOS, ICU status, diagnoses, lab summary, pre-admission vitals, and identified care gaps.

---

## FHIR Export (`build_fhir.py`)

Each patient is exported as a **FHIR R4 Transaction Bundle** (one JSON file per patient).

Resources per bundle:

| Resource | Content |
|---|---|
| Patient | Gender, birth year, deceased date |
| Encounter | Admission type, period, discharge disposition |
| Condition | One per ICD code (ICD-9 and ICD-10 auto-detected) |
| Observation | Pre-admission OMR vitals with LOINC codes |
| MedicationRequest | One per prescribed drug |
| Procedure | One per procedure event |

Bundles use `urn:uuid` internal references and are ready for `POST /fhir` to a HAPI FHIR server.

---

## File Structure

```
final/
├── app.py                  # Streamlit dashboard
├── care_gap_engine.py      # Rule-based care gap detection
├── build_profiles.py       # ETL: raw MIMIC-IV → processed CSVs + JSONL
├── build_subset.py         # Select 1,000 clinically rich patients
├── build_fhir.py           # Convert subset to FHIR Transaction Bundles
├── rebuild_patient_dict.py # Fast rebuild of patient_dict.jsonl only
├── dataset/                # Raw MIMIC-IV CSV files (not tracked)
├── processed/              # Processed output files (not tracked)
└── fhir_output/            # FHIR JSON bundles (not tracked)
```

---

## Requirements

```bash
pip install streamlit plotly pandas anthropic
```

> `anthropic` is optional — only needed if connecting a Claude-based LLM backend.
