import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path('./dataset/')
OUT_DIR  = Path('./processed/')
OUT_DIR.mkdir(exist_ok=True)

CHUNK = 500_000

print("=" * 60)
print("MIMIC-IV 데이터 프로세싱 시작")
print("=" * 60)

# ────────────────────────────────────────────────────────────
# 1. 작은 파일 로드
# ────────────────────────────────────────────────────────────
print("\n▶ [1/8] 기본 파일 로드 중...")

patients        = pd.read_csv(DATA_DIR / 'patients.csv')
admissions      = pd.read_csv(DATA_DIR / 'admissions.csv',
                               parse_dates=['admittime', 'dischtime'])
icustays        = pd.read_csv(DATA_DIR / 'icustays.csv')
d_items         = pd.read_csv(DATA_DIR / 'd_items.csv')
d_labitems      = pd.read_csv(DATA_DIR / 'd_labitems.csv')
diagnoses_icd   = pd.read_csv(DATA_DIR / 'diagnoses_icd.csv',
                               dtype={'icd_code': str})
d_icd_diagnoses = pd.read_csv(DATA_DIR / 'd_icd_diagnoses.csv',
                               dtype={'icd_code': str})

admissions['los_hours'] = (
    (admissions['dischtime'] - admissions['admittime'])
    .dt.total_seconds() / 3600
)
print("  완료")

# ────────────────────────────────────────────────────────────
# 2. 진단 집계  →  hadm_id 기준
# ────────────────────────────────────────────────────────────
print("\n▶ [2/8] 진단 집계 중...")

diagnoses = diagnoses_icd.merge(
    d_icd_diagnoses[['icd_code', 'icd_version', 'long_title']],
    on=['icd_code', 'icd_version'], how='left'
)

dx_per_adm = (
    diagnoses.sort_values(['hadm_id', 'seq_num'])
    .groupby('hadm_id')
    .agg(
        n_diagnoses     = ('icd_code',   'count'),
        primary_dx_code = ('icd_code',   'first'),
        primary_dx_name = ('long_title', 'first'),
        diagnoses_list  = ('long_title', lambda x: ' | '.join(x.dropna())),
        icd_codes_list  = ('icd_code',   lambda x: ' | '.join(x)),
    )
    .reset_index()
)
print(f"  완료: {len(dx_per_adm):,} 입원")

# ────────────────────────────────────────────────────────────
# 3. ICU 집계  →  hadm_id 기준
# ────────────────────────────────────────────────────────────
print("\n▶ [3/8] ICU 집계 중...")

icu_per_adm = (
    icustays.groupby('hadm_id')
    .agg(
        n_icu_stays   = ('stay_id',        'count'),
        total_icu_los = ('los',            'sum'),
        care_units    = ('first_careunit', lambda x: ' | '.join(x.unique())),
    )
    .reset_index()
)
icu_per_adm['has_icu'] = True
print(f"  완료: {len(icu_per_adm):,} 입원")

# ────────────────────────────────────────────────────────────
# 4. 시술 집계  →  hadm_id 기준
# ────────────────────────────────────────────────────────────
print("\n▶ [4/8] 시술 집계 중... (144MB)")

procedureevents = pd.read_csv(DATA_DIR / 'procedureevents.csv')
proc_named = procedureevents.merge(
    d_items[['itemid', 'label', 'category']], on='itemid', how='left'
)
proc_per_adm = (
    proc_named.groupby('hadm_id')
    .agg(
        n_procedures         = ('itemid',   'count'),
        procedure_categories = ('category', lambda x: ' | '.join(x.dropna().unique())),
        procedure_labels     = ('label',    lambda x: ' | '.join(x.dropna().unique())),
    )
    .reset_index()
)
print(f"  완료: {len(proc_per_adm):,} 입원")

# ────────────────────────────────────────────────────────────
# 5. labevents 집계  →  hadm_id 기준  (18GB → 청크 처리)
# ────────────────────────────────────────────────────────────
print("\n▶ [5/8] labevents 집계 중... (18GB, 가장 오래 걸림)")

lab_acc = {}   # hadm_id → {n, labels, n_abn}

reader = pd.read_csv(
    DATA_DIR / 'labevents.csv',
    usecols=['hadm_id', 'itemid', 'flag'],
    dtype={'hadm_id': 'Int64'},
    chunksize=CHUNK
)

for i, chunk in enumerate(reader, 1):
    chunk = chunk.dropna(subset=['hadm_id'])
    chunk = chunk.assign(hadm_id=chunk['hadm_id'].astype(int))
    chunk = chunk.merge(d_labitems[['itemid', 'label']], on='itemid', how='left')

    grp = chunk.groupby('hadm_id')
    for hid, g in grp:
        if hid not in lab_acc:
            lab_acc[hid] = {'n': 0, 'labels': set(), 'n_abn': 0}
        lab_acc[hid]['n']      += len(g)
        lab_acc[hid]['labels'] |= set(g['label'].dropna())
        lab_acc[hid]['n_abn']  += int(g['flag'].notna().sum())

    if i % 20 == 0:
        print(f"  청크 {i} 완료...")

lab_per_adm = pd.DataFrame([
    {
        'hadm_id':        hid,
        'n_labs':         v['n'],
        'lab_tests_done': ' | '.join(sorted(v['labels'])),
        'n_abnormal_labs':v['n_abn'],
    }
    for hid, v in lab_acc.items()
])
print(f"  완료: {len(lab_per_adm):,} 입원")

# ────────────────────────────────────────────────────────────
# 6. prescriptions 집계  →  hadm_id 기준  (3.3GB → 청크 처리)
# ────────────────────────────────────────────────────────────
print("\n▶ [6/8] prescriptions 집계 중... (3.3GB)")

rx_agg_chunks = []
for chunk in pd.read_csv(
    DATA_DIR / 'prescriptions.csv',
    usecols=['subject_id', 'hadm_id', 'drug'],
    chunksize=CHUNK
):
    chunk = chunk.dropna(subset=['hadm_id'])
    rx_agg_chunks.append(
        chunk.groupby('hadm_id')
        .agg(n_prescriptions=('drug', 'count'),
             drugs_list=('drug', lambda x: ' | '.join(x.dropna().unique())))
        .reset_index()
    )

rx_per_adm = (
    pd.concat(rx_agg_chunks, ignore_index=True)
    .groupby('hadm_id')
    .agg(n_prescriptions=('n_prescriptions', 'sum'),
         drugs_list=('drugs_list', lambda x: ' | '.join(x)))
    .reset_index()
)
print(f"  완료: {len(rx_per_adm):,} 입원")

# ────────────────────────────────────────────────────────────
# 7. omr 집계  →  subject_id 기준  (hadm_id 없음)
# ────────────────────────────────────────────────────────────
print("\n▶ [7/8] omr 집계 중... (307MB)")

omr = pd.read_csv(DATA_DIR / 'omr.csv', parse_dates=['chartdate'])

omr_per_patient = (
    omr.groupby('subject_id')
    .agg(
        n_observations    = ('result_name', 'count'),
        observation_types = ('result_name', lambda x: ' | '.join(x.unique())),
    )
    .reset_index()
)

# 최근값 (대시보드 표시용)
bp_latest = (
    omr[omr['result_name'] == 'Blood Pressure']
    .sort_values('chartdate')
    .groupby('subject_id')['result_value']
    .last()
    .rename('bp_latest')
)
bmi_latest = (
    omr[omr['result_name'] == 'BMI (kg/m2)']
    .sort_values('chartdate')
    .groupby('subject_id')['result_value']
    .last()
    .rename('bmi_latest')
)

omr_per_patient = (
    omr_per_patient
    .merge(bp_latest,  on='subject_id', how='left')
    .merge(bmi_latest, on='subject_id', how='left')
)
print(f"  완료: {len(omr_per_patient):,} 환자")

print("  omr lookup 빌드 중...")
epoch = pd.Timestamp('1970-01-01')
omr_lookup = defaultdict(list)
for row in omr[['subject_id', 'chartdate', 'result_name', 'result_value']].itertuples(index=False):
    omr_lookup[row.subject_id].append({
        'date_days': (pd.to_datetime(row.chartdate) - epoch).days,
        'name':  row.result_name,
        'value': str(row.result_value),
    })
print(f"  lookup 완료: {len(omr_lookup):,} 환자")

# ────────────────────────────────────────────────────────────
# 8-A. admission_profile.csv 빌드
# ────────────────────────────────────────────────────────────
print("\n▶ [8/8] 파일 생성 중...")
print("  admission_profile.csv 빌드...")

ADM_COLS = [
    'subject_id', 'hadm_id', 'admittime', 'dischtime', 'los_hours',
    'admission_type', 'hospital_expire_flag', 'discharge_location',
    'insurance', 'race', 'marital_status',
]

admission_profile = (
    admissions[ADM_COLS]
    .merge(patients[['subject_id', 'gender', 'anchor_age']], on='subject_id', how='left')
    .merge(dx_per_adm,   on='hadm_id', how='left')
    .merge(icu_per_adm,  on='hadm_id', how='left')
    .merge(proc_per_adm, on='hadm_id', how='left')
    .merge(lab_per_adm,  on='hadm_id', how='left')
    .merge(rx_per_adm,   on='hadm_id', how='left')
)

for col, default, dtype in [
    ('has_icu',          False, bool),
    ('n_icu_stays',      0,     int),
    ('total_icu_los',    0.0,   float),
    ('n_procedures',     0,     int),
    ('n_diagnoses',      0,     int),
    ('n_labs',           0,     int),
    ('n_abnormal_labs',  0,     int),
    ('n_prescriptions',  0,     int),
]:
    admission_profile[col] = admission_profile[col].fillna(default).astype(dtype)

admission_profile.to_csv(OUT_DIR / 'admission_profile.csv', index=False)
print(f"  저장 완료: processed/admission_profile.csv  {admission_profile.shape}")

# ────────────────────────────────────────────────────────────
# 8-B. patient_summary.csv 빌드
# ────────────────────────────────────────────────────────────
print("  patient_summary.csv 빌드...")

adm_stats = (
    admissions.groupby('subject_id')
    .agg(
        n_admissions      = ('hadm_id',             'count'),
        n_hospital_deaths = ('hospital_expire_flag', 'sum'),
        avg_los_hours     = ('los_hours',            'mean'),
    )
    .reset_index()
)
icu_stats = (
    icustays.groupby('subject_id')
    .agg(
        total_icu_stays = ('stay_id', 'count'),
        total_icu_los   = ('los',     'sum'),
    )
    .reset_index()
)
dx_stats = (
    diagnoses.groupby('subject_id')
    .agg(
        all_diagnoses = ('long_title', lambda x: ' | '.join(x.dropna().unique()[:15])),
        all_icd_codes = ('icd_code',   lambda x: ' | '.join(x.unique()[:15])),
    )
    .reset_index()
)

patient_summary = (
    patients
    .merge(adm_stats,       on='subject_id', how='left')
    .merge(icu_stats,       on='subject_id', how='left')
    .merge(dx_stats,        on='subject_id', how='left')
    .merge(omr_per_patient, on='subject_id', how='left')
)

patient_summary.to_csv(OUT_DIR / 'patient_summary.csv', index=False)
print(f"  저장 완료: processed/patient_summary.csv    {patient_summary.shape}")

# ────────────────────────────────────────────────────────────
# 8-C. patient_dict.jsonl 빌드
# ────────────────────────────────────────────────────────────
print("  patient_dict.jsonl 빌드...")

ADM_DICT_COLS = [
    'hadm_id', 'admittime', 'dischtime', 'los_hours', 'admission_type',
    'hospital_expire_flag',
    'primary_dx_code', 'primary_dx_name', 'diagnoses_list', 'n_diagnoses',
    'icd_codes_list',
    'has_icu', 'n_icu_stays', 'total_icu_los', 'care_units',
    'n_procedures', 'procedure_categories', 'procedure_labels',
    'n_labs', 'lab_tests_done', 'n_abnormal_labs',
    'n_prescriptions', 'drugs_list',
]

pt_info  = patients.set_index('subject_id').to_dict('index')

out_path = OUT_DIR / 'patient_dict.jsonl'
n_written = 0

with open(out_path, 'w', encoding='utf-8') as f:
    for subject_id, grp in admission_profile.groupby('subject_id'):
        sid = int(subject_id)
        patient_omr = omr_lookup.get(sid, [])

        admissions_list = []
        for _, adm_row in grp.iterrows():
            admittime_days = (pd.to_datetime(adm_row['admittime']) - epoch).days

            pre_omr = []
            for rec in patient_omr:
                days_before = admittime_days - rec['date_days']
                if days_before >= 0:
                    pre_omr.append({
                        'days_before': int(days_before),
                        'name':  rec['name'],
                        'value': rec['value'],
                    })
            pre_omr.sort(key=lambda x: x['days_before'])

            adm_dict = {
                c: ('' if pd.isna(adm_row[c]) else adm_row[c])
                for c in ADM_DICT_COLS if c in adm_row.index
            }
            adm_dict['admittime']         = str(adm_row['admittime'])
            adm_dict['dischtime']         = str(adm_row['dischtime'])
            adm_dict['pre_admission_omr'] = pre_omr[:20]  # 최근 20건만
            admissions_list.append(adm_dict)

        record = {
            'subject_id': sid,
            'info':       pt_info.get(sid, {}),
            'admissions': admissions_list,
        }
        f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
        n_written += 1

print(f"  저장 완료: processed/patient_dict.jsonl     {n_written:,} 환자")

# ────────────────────────────────────────────────────────────
# 완료 요약
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("처리 완료!")
print(f"  admission_profile.csv : {admission_profile.shape[0]:,}행 × {admission_profile.shape[1]}열")
print(f"  patient_summary.csv   : {patient_summary.shape[0]:,}행 × {patient_summary.shape[1]}열")
print(f"  patient_dict.jsonl    : {n_written:,}명")
print("=" * 60)
