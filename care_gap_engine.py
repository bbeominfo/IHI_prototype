"""
Rule-based Care Gap Engine
Takes one admission dict and returns a list of care gaps.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Gap:
    rule_id:  str
    severity: str   # HIGH / MEDIUM / LOW
    category: str
    message:  str


# ── Helpers ───────────────────────────────────────────────────

def _has_dx(admission: dict, *keywords: str) -> bool:
    dx  = str(admission.get('diagnoses_list', '')).lower()
    icd = str(admission.get('icd_codes_list', '')).lower()
    return any(kw.lower() in dx or kw.lower() in icd for kw in keywords)

def _has_lab(admission: dict, *keywords: str) -> bool:
    labs = str(admission.get('lab_tests_done', '')).lower()
    return any(kw.lower() in labs for kw in keywords)

def _has_drug(admission: dict, *keywords: str) -> bool:
    drugs = str(admission.get('drugs_list', '')).lower()
    return any(kw.lower() in drugs for kw in keywords)

def _parse_bp(value: str) -> tuple[int, int] | None:
    """'120/80' → (120, 80), returns None on parse failure"""
    try:
        parts = str(value).strip().split('/')
        return int(parts[0]), int(parts[1])
    except Exception:
        return None

def _omr_values(admission: dict, name_keyword: str) -> list[str]:
    """Return value list for a given OMR measure from pre_admission_omr"""
    return [
        r['value']
        for r in admission.get('pre_admission_omr', [])
        if name_keyword.lower() in r.get('name', '').lower()
    ]


# ── Rules ─────────────────────────────────────────────────────

def _r001(adm: dict) -> Gap | None:
    """Diabetes + no HbA1c"""
    if _has_dx(adm, 'diabetes', 'diabetic', 'E11', 'E10', 'E12', 'E13', '250'):
        if not _has_lab(adm, 'hemoglobin a1c', 'hba1c', 'a1c'):
            return Gap('R001', 'HIGH', 'Missing Test',
                       'Diabetes diagnosis present but HbA1c was not measured during this admission.')
    return None

def _r002(adm: dict) -> Gap | None:
    """Diabetes + no glucose"""
    if _has_dx(adm, 'diabetes', 'diabetic', 'E11', 'E10', '250'):
        if not _has_lab(adm, 'glucose'):
            return Gap('R002', 'MEDIUM', 'Missing Test',
                       'Diabetes diagnosis present but blood glucose was not measured during this admission.')
    return None

def _r003(adm: dict) -> Gap | None:
    """Renal failure + no creatinine"""
    if _has_dx(adm, 'renal failure', 'kidney failure', 'chronic kidney',
               'renal insufficiency', 'N17', 'N18', 'N19', '585', '586'):
        if not _has_lab(adm, 'creatinine'):
            return Gap('R003', 'HIGH', 'Missing Test',
                       'Renal failure diagnosis present but creatinine was not measured during this admission.')
    return None

def _r004(adm: dict) -> Gap | None:
    """Liver disease + no ALT/AST"""
    if _has_dx(adm, 'cirrhosis', 'hepatitis', 'liver disease',
               'hepatic', 'K70', 'K71', 'K72', 'K73', 'K74', '571', '572', '573'):
        if not _has_lab(adm, 'alanine aminotransferase', 'alt',
                         'asparate aminotransferase', 'ast'):
            return Gap('R004', 'HIGH', 'Missing Test',
                       'Liver disease diagnosis present but liver function tests (ALT/AST) were not performed.')
    return None

def _r005(adm: dict) -> Gap | None:
    """HIV + no CD4 count"""
    if _has_dx(adm, 'hiv', 'human immunodeficiency', 'B20', 'Z21', '042', 'V08'):
        if not _has_lab(adm, 'cd4', 'absolute cd4'):
            return Gap('R005', 'MEDIUM', 'Missing Test',
                       'HIV diagnosis present but CD4 count was not measured during this admission.')
    return None

def _r006(adm: dict) -> Gap | None:
    """Heart failure + no BNP"""
    if _has_dx(adm, 'heart failure', 'cardiac failure', 'I50', '428'):
        if not _has_lab(adm, 'bnp', 'brain natriuretic', 'proBNP', 'NT-proBNP'):
            return Gap('R006', 'MEDIUM', 'Missing Test',
                       'Heart failure diagnosis present but BNP was not measured during this admission.')
    return None

def _r007(adm: dict) -> Gap | None:
    """Metformin + no creatinine (nephrotoxicity safety)"""
    if _has_drug(adm, 'metformin'):
        if not _has_lab(adm, 'creatinine'):
            return Gap('R007', 'HIGH', 'Drug Safety',
                       'Metformin prescribed without creatinine monitoring. '
                       'Impaired renal function increases risk of lactic acidosis.')
    return None

def _r008(adm: dict) -> Gap | None:
    """Anticoagulant + no PT/INR"""
    if _has_drug(adm, 'warfarin', 'coumadin', 'heparin'):
        if not _has_lab(adm, 'inr', 'pt ', 'prothrombin'):
            return Gap('R008', 'HIGH', 'Drug Safety',
                       'Anticoagulant prescribed without PT/INR monitoring. Bleeding risk assessment required.')
    return None

def _r009(adm: dict) -> Gap | None:
    """BP > 140/90 on 2+ outpatient visits"""
    bp_values = _omr_values(adm, 'blood pressure')
    high_count = sum(
        1 for v in bp_values
        if (parsed := _parse_bp(v)) and (parsed[0] > 140 or parsed[1] > 90)
    )
    if high_count >= 2:
        return Gap('R009', 'MEDIUM', 'Vital Sign Alert',
                   f'Blood pressure exceeded 140/90 on {high_count} pre-admission outpatient visits. '
                   'Hypertension control should be reviewed.')
    return None

def _r010(adm: dict) -> Gap | None:
    """BMI < 18.5 (underweight / malnutrition)"""
    for v in _omr_values(adm, 'bmi'):
        try:
            if float(v) < 18.5:
                return Gap('R010', 'MEDIUM', 'Vital Sign Alert',
                           f'Pre-admission BMI of {v} is below 18.5 (underweight threshold). '
                           'Nutritional assessment recommended.')
        except Exception:
            continue
    return None

def _r011(adm: dict) -> Gap | None:
    """Weight loss trend (recent vs. past)"""
    weights = []
    for v in _omr_values(adm, 'weight'):
        try:
            weights.append(float(str(v).replace(',', '')))
        except Exception:
            continue
    # pre_admission_omr sorted ascending by days_before → index 0 is most recent
    if len(weights) >= 3:
        recent = sum(weights[:2]) / 2
        past   = sum(weights[-2:]) / 2
        loss_pct = (past - recent) / past * 100
        if loss_pct >= 5:
            return Gap('R011', 'LOW', 'Vital Sign Alert',
                       f'Pre-admission weight declined by approximately {loss_pct:.1f}%. '
                       'If unintentional, further evaluation is warranted.')
    return None

def _r012(adm: dict) -> Gap | None:
    """Abnormal lab ratio > 50% (minimum 10 labs)"""
    n_labs = adm.get('n_labs', 0)
    n_abn  = adm.get('n_abnormal_labs', 0)
    if n_labs >= 10 and n_abn / n_labs > 0.5:
        pct = n_abn / n_labs * 100
        return Gap('R012', 'HIGH', 'Lab Alert',
                   f'{n_abn} of {n_labs} lab results ({pct:.0f}%) were abnormal. '
                   'Comprehensive clinical reassessment is recommended.')
    return None


# ── Public API ────────────────────────────────────────────────

RULES = [_r001, _r002, _r003, _r004, _r005, _r006,
         _r007, _r008, _r009, _r010, _r011, _r012]

SEVERITY_ORDER = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}

def check_care_gaps(admission: dict) -> list[Gap]:
    """Return care gaps for an admission dict, sorted by severity."""
    gaps = [rule(admission) for rule in RULES]
    gaps = [g for g in gaps if g is not None]
    gaps.sort(key=lambda g: SEVERITY_ORDER.get(g.severity, 9))
    return gaps


def format_gaps(gaps: list[Gap]) -> str:
    """Convert gap list to plain text for LLM prompt context."""
    if not gaps:
        return 'No care gaps identified.'
    return '\n'.join(f'[{g.severity}] ({g.category}) {g.message}' for g in gaps)
