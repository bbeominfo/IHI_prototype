"""
Clinical Decision Support Dashboard
MIMIC-IV subset (1,000 patients) — Streamlit
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from care_gap_engine import check_care_gaps, format_gaps, Gap

# ── Config ────────────────────────────────────────────────────
SUBSET = Path('./processed/subset/')
SEVERITY_COLOR = {'HIGH': '#d32f2f', 'MEDIUM': '#f57c00', 'LOW': '#388e3c'}
SEVERITY_BG    = {'HIGH': '#ffebee', 'MEDIUM': '#fff3e0', 'LOW': '#e8f5e9'}

st.set_page_config(
    page_title='Clinical Dashboard',
    page_icon='🏥',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
.gap-card {
    border-left: 5px solid;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 14px;
}
.metric-box {
    background: #f5f5f5;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
.section-header {
    font-size: 16px;
    font-weight: 600;
    color: #1a237e;
    border-bottom: 2px solid #e8eaf6;
    padding-bottom: 4px;
    margin: 16px 0 10px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────
@st.cache_data
def load_index() -> dict[int, int]:
    """subject_id → byte offset in JSONL"""
    index = {}
    path = SUBSET / 'patient_dict.jsonl'
    with open(path, 'rb') as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            sid = json.loads(line)['subject_id']
            index[sid] = offset
    return index

@st.cache_data
def load_patient(subject_id: int) -> dict | None:
    index = load_index()
    if subject_id not in index:
        return None
    path = SUBSET / 'patient_dict.jsonl'
    with open(path, 'rb') as f:
        f.seek(index[subject_id])
        return json.loads(f.readline())

@st.cache_data
def all_subject_ids() -> list[int]:
    return sorted(load_index().keys())


# ── LLM prompt builder (model-agnostic) ───────────────────────
def build_prompt(patient: dict, adm: dict, gaps: list[Gap]) -> str:
    info = patient.get('info', {})
    age  = info.get('anchor_age', '?')
    sex  = 'Female' if info.get('gender') == 'F' else 'Male'
    los  = round(adm.get('los_hours', 0) / 24, 1)

    dx_list   = str(adm.get('diagnoses_list', '')).replace('|', '\n  -')
    drug_list = ', '.join(str(adm.get('drugs_list', '')).split(' | ')[:10])
    n_labs    = adm.get('n_labs', 0)
    n_abn     = adm.get('n_abnormal_labs', 0)
    omr_str   = '\n'.join(
        f"  {r['days_before']}d prior: {r['name']} = {r['value']}"
        for r in adm.get('pre_admission_omr', [])[:8]
    )
    gap_str = format_gaps(gaps) if gaps else 'None detected'

    return f"""You are a clinical decision support assistant summarizing a hospital admission for a physician.

Patient: {sex}, age {age}
Admission type: {adm.get('admission_type', 'Unknown')}
Length of stay: {los} days
ICU: {'Yes' if adm.get('has_icu') else 'No'}

Primary diagnosis: {adm.get('primary_dx_name', 'Unknown')}
All diagnoses:
  -{dx_list}

Laboratory: {n_labs} tests, {n_abn} abnormal ({round(n_abn/n_labs*100) if n_labs else 0}%)
Medications (sample): {drug_list}

Pre-admission vitals (OMR):
{omr_str if omr_str else '  None recorded'}

Identified care gaps:
{gap_str}

Write a concise clinical summary (3–4 sentences) followed by a brief explanation of each care gap and its clinical importance. Use plain, physician-friendly language. Do not repeat the raw data verbatim."""


# ── Chart helpers ──────────────────────────────────────────────
def lab_donut(n_labs: int, n_abn: int) -> go.Figure:
    normal = max(n_labs - n_abn, 0)
    fig = go.Figure(go.Pie(
        labels=['Normal', 'Abnormal'],
        values=[normal, n_abn],
        hole=0.55,
        marker_colors=['#4caf50', '#f44336'],
        textinfo='percent+label',
        hovertemplate='%{label}: %{value}<extra></extra>',
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
        height=200,
        annotations=[dict(text=f'{n_labs}<br>labs', x=0.5, y=0.5,
                          font_size=14, showarrow=False)],
    )
    return fig


def vitals_timeline(omr_records: list[dict]) -> go.Figure | None:
    rows = []
    for r in omr_records:
        name = r.get('name', '')
        val  = r.get('value', '')
        days = r.get('days_before', 0)
        # try to extract numeric from BP (systolic only)
        if 'Blood Pressure' in name:
            try:
                sys_val = float(str(val).split('/')[0])
                rows.append({'days_before': days, 'measure': 'BP Systolic', 'value': sys_val})
            except Exception:
                pass
        else:
            try:
                rows.append({'days_before': days, 'measure': name, 'value': float(val)})
            except Exception:
                pass

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values('days_before', ascending=False)
    measures = df['measure'].unique()
    if len(measures) > 5:
        measures = measures[:5]
    df = df[df['measure'].isin(measures)]

    fig = px.line(
        df, x='days_before', y='value', color='measure',
        labels={'days_before': 'Days before admission', 'value': 'Value'},
        markers=True,
    )
    fig.update_xaxes(autorange='reversed', title='Days before admission')
    fig.update_layout(
        height=280,
        margin=dict(t=10, b=40, l=40, r=10),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    return fig


def admissions_timeline(admissions: list[dict]) -> go.Figure:
    rows = []
    for adm in admissions:
        try:
            start = pd.to_datetime(adm['admittime'])
            end   = pd.to_datetime(adm['dischtime'])
        except Exception:
            continue
        rows.append({
            'hadm_id':   str(adm.get('hadm_id', '')),
            'Start':     start,
            'End':       end,
            'Type':      adm.get('admission_type', 'Unknown'),
            'ICU':       '✓ ICU' if adm.get('has_icu') else 'No ICU',
            'Primary Dx':adm.get('primary_dx_name', ''),
        })

    if not rows:
        return go.Figure()

    df = pd.DataFrame(rows)
    fig = px.timeline(
        df, x_start='Start', x_end='End', y='hadm_id',
        color='Type',
        hover_data=['Primary Dx', 'ICU'],
        labels={'hadm_id': 'Admission'},
    )
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(
        height=max(150, 40 * len(rows)),
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
    )
    return fig


# ── Gap rendering ──────────────────────────────────────────────
def render_gaps(gaps: list[Gap]):
    if not gaps:
        st.success('No care gaps identified for this admission.')
        return
    for g in gaps:
        color  = SEVERITY_COLOR.get(g.severity, '#888')
        bg     = SEVERITY_BG.get(g.severity, '#fafafa')
        badge  = f'<b style="color:{color}">[{g.severity}]</b>'
        cat    = f'<span style="color:#555;font-size:12px">({g.category})</span>'
        st.markdown(
            f'<div class="gap-card" style="border-color:{color};background:{bg}">'
            f'{badge} {cat}<br>{g.message}</div>',
            unsafe_allow_html=True,
        )


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.title('🏥 Clinical Dashboard')
    st.caption('MIMIC-IV · 1,000 patient subset')
    st.divider()

    ids = all_subject_ids()
    st.markdown(f'**{len(ids):,} patients loaded**')

    search_input = st.text_input('Search by Subject ID', placeholder='e.g. 10011938')
    if search_input.strip().isdigit():
        query_id = int(search_input.strip())
        if query_id in set(ids):
            selected_id = query_id
        else:
            st.warning('Subject ID not found in subset.')
            selected_id = ids[0]
    else:
        selected_id = st.selectbox('Or select a patient', ids, index=0)

    st.divider()
    st.caption('LLM integration coming soon.')


# ── Main panel ────────────────────────────────────────────────
patient = load_patient(selected_id)
if patient is None:
    st.error('Patient not found.')
    st.stop()

info       = patient.get('info', {})
admissions = patient.get('admissions', [])
admissions = sorted(admissions, key=lambda a: str(a.get('admittime', '')))

# Patient header
age   = info.get('anchor_age', '?')
sex   = '♀ Female' if info.get('gender') == 'F' else '♂ Male'
dod   = info.get('dod', '')
alive = '' if not dod or str(dod) in ('', 'nan', 'NaT') else f' · Deceased {dod}'

st.header(f'Patient #{selected_id}')
st.caption(f'{sex} · Age at anchor year: {age}{alive}')

# Admission timeline
st.markdown('<div class="section-header">Admission Timeline</div>', unsafe_allow_html=True)
st.plotly_chart(admissions_timeline(admissions), use_container_width=True)

# Admission selector
adm_labels = [
    f"[{i+1}] {a.get('admittime','')[:10]}  —  {a.get('primary_dx_name','Unknown')[:40]}"
    for i, a in enumerate(admissions)
]
adm_idx = st.selectbox('Select admission', range(len(admissions)),
                        format_func=lambda i: adm_labels[i])
adm = admissions[adm_idx]

# Precompute care gaps
gaps = check_care_gaps(adm)

st.divider()

# ── Top metrics ───────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
los_days = round(adm.get('los_hours', 0) / 24, 1)
c1.metric('LOS', f'{los_days}d')
c2.metric('Diagnoses', adm.get('n_diagnoses', 0))
c3.metric('Lab tests', adm.get('n_labs', 0))
c4.metric('Abnormal labs', adm.get('n_abnormal_labs', 0))
c5.metric('Medications', adm.get('n_prescriptions', 0))
c6.metric('Care gaps', len(gaps), delta=None)

# ── Tabs ──────────────────────────────────────────────────────
tab_gaps, tab_labs, tab_dx, tab_vitals, tab_meds, tab_llm = st.tabs([
    f'⚠️ Care Gaps ({len(gaps)})', '🧪 Labs', '📋 Diagnoses',
    '📈 Pre-admission Vitals', '💊 Medications', '🤖 AI Summary',
])

# ── Tab: Care Gaps ────────────────────────────────────────────
with tab_gaps:
    st.markdown('<div class="section-header">Identified Care Gaps</div>',
                unsafe_allow_html=True)
    render_gaps(gaps)

    if gaps:
        sev_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for g in gaps:
            sev_counts[g.severity] = sev_counts.get(g.severity, 0) + 1
        fig_sev = px.bar(
            x=list(sev_counts.keys()),
            y=list(sev_counts.values()),
            color=list(sev_counts.keys()),
            color_discrete_map={k: SEVERITY_COLOR[k] for k in sev_counts},
            labels={'x': 'Severity', 'y': 'Count'},
            height=220,
        )
        fig_sev.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_sev, use_container_width=True)

# ── Tab: Labs ─────────────────────────────────────────────────
with tab_labs:
    cola, colb = st.columns([1, 2])
    with cola:
        st.markdown('<div class="section-header">Lab Result Summary</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(lab_donut(adm.get('n_labs', 0), adm.get('n_abnormal_labs', 0)),
                        use_container_width=True)
        abn_pct = (adm.get('n_abnormal_labs', 0) / adm.get('n_labs', 1)) * 100
        if abn_pct > 50:
            st.error(f'{abn_pct:.0f}% abnormal — critical review needed')
        elif abn_pct > 30:
            st.warning(f'{abn_pct:.0f}% abnormal')
        else:
            st.success(f'{abn_pct:.0f}% abnormal')

    with colb:
        st.markdown('<div class="section-header">Tests Performed</div>',
                    unsafe_allow_html=True)
        lab_list = [t.strip() for t in str(adm.get('lab_tests_done', '')).split('|') if t.strip()]
        if lab_list:
            cols = st.columns(2)
            half = len(lab_list) // 2 + len(lab_list) % 2
            with cols[0]:
                for t in lab_list[:half]:
                    st.caption(f'• {t}')
            with cols[1]:
                for t in lab_list[half:]:
                    st.caption(f'• {t}')
        else:
            st.caption('No lab tests recorded.')

# ── Tab: Diagnoses ────────────────────────────────────────────
with tab_dx:
    st.markdown('<div class="section-header">Diagnoses</div>', unsafe_allow_html=True)
    dx_names = [d.strip() for d in str(adm.get('diagnoses_list', '')).split('|') if d.strip()]
    icd_codes = [c.strip() for c in str(adm.get('icd_codes_list', '')).split('|') if c.strip()]

    st.caption(f'Primary: **{adm.get("primary_dx_name", "Unknown")}** '
               f'({adm.get("primary_dx_code", "")})')
    st.markdown('---')

    for i, (dx, icd) in enumerate(zip(dx_names, icd_codes), 1):
        col_n, col_c = st.columns([4, 1])
        col_n.write(f'{i}. {dx}')
        col_c.caption(icd)

# ── Tab: Pre-admission Vitals ──────────────────────────────────
with tab_vitals:
    omr = adm.get('pre_admission_omr', [])
    st.markdown('<div class="section-header">Pre-admission OMR Records</div>',
                unsafe_allow_html=True)

    if omr:
        omr_df = pd.DataFrame(omr).rename(
            columns={'days_before': 'Days before admission', 'name': 'Measure', 'value': 'Value'})
        st.dataframe(omr_df, use_container_width=True, hide_index=True)

        fig_v = vitals_timeline(omr)
        if fig_v:
            st.markdown('<div class="section-header">Trend</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_v, use_container_width=True)
    else:
        st.info('No pre-admission OMR records for this admission.')

# ── Tab: Medications ──────────────────────────────────────────
with tab_meds:
    st.markdown('<div class="section-header">Prescribed Medications</div>',
                unsafe_allow_html=True)
    drug_list = [d.strip() for d in str(adm.get('drugs_list', '')).split('|') if d.strip()]
    if drug_list:
        cols = st.columns(2)
        half = len(drug_list) // 2 + len(drug_list) % 2
        with cols[0]:
            for d in drug_list[:half]:
                st.write(f'• {d}')
        with cols[1]:
            for d in drug_list[half:]:
                st.write(f'• {d}')
    else:
        st.info('No medication records.')

    procs = [p.strip() for p in str(adm.get('procedure_labels', '')).split('|') if p.strip()]
    if procs:
        st.markdown('<div class="section-header">Procedures</div>', unsafe_allow_html=True)
        for p in procs:
            st.write(f'• {p}')

# ── Tab: AI Summary ───────────────────────────────────────────
with tab_llm:
    st.markdown('<div class="section-header">AI Clinical Summary</div>',
                unsafe_allow_html=True)
    st.info('LLM integration coming soon. The prompt below will be passed to the model.')
    st.markdown('**Prompt that will be sent to the LLM:**')
    st.code(build_prompt(patient, adm, gaps), language='text')
