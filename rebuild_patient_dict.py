import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path('./dataset/')
OUT_DIR  = Path('./processed/')

print("=" * 60)
print("patient_dict.jsonl 재빌드 (빠른 버전)")
print("=" * 60)

# ── 이미 만들어진 파일 활용 ──────────────────────────────────
print("\n▶ admission_profile.csv 로드 중... (923MB)")
admission_profile = pd.read_csv(
    OUT_DIR / 'admission_profile.csv',
    parse_dates=['admittime', 'dischtime']
)
print(f"  완료: {admission_profile.shape}")

print("\n▶ patients.csv 로드 중...")
patients = pd.read_csv(DATA_DIR / 'patients.csv')
pt_info  = patients.set_index('subject_id').to_dict('index')
print(f"  완료: {len(pt_info):,} 환자")

# ── omr lookup 빌드 ───────────────────────────────────────────
print("\n▶ omr.csv 로드 및 lookup 빌드 중... (307MB)")
omr = pd.read_csv(DATA_DIR / 'omr.csv', parse_dates=['chartdate'])

epoch = pd.Timestamp('1970-01-01')
omr_lookup = defaultdict(list)
for row in omr[['subject_id', 'chartdate', 'result_name', 'result_value']].itertuples(index=False):
    omr_lookup[row.subject_id].append({
        'date_days': (pd.to_datetime(row.chartdate) - epoch).days,
        'name':  row.result_name,
        'value': str(row.result_value),
    })
print(f"  완료: {len(omr_lookup):,} 환자")

# ── patient_dict.jsonl 빌드 ───────────────────────────────────
print("\n▶ patient_dict.jsonl 빌드 중...")

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
            adm_dict['pre_admission_omr'] = pre_omr[:20]  # 최근 20건
            admissions_list.append(adm_dict)

        record = {
            'subject_id': sid,
            'info':       pt_info.get(sid, {}),
            'admissions': admissions_list,
        }
        f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
        n_written += 1

        if n_written % 50000 == 0:
            print(f"  {n_written:,}명 처리됨...")

print(f"\n저장 완료: processed/patient_dict.jsonl  {n_written:,}명")
print("=" * 60)

# ── 샘플 확인 ────────────────────────────────────────────────
print("\n▶ 샘플 확인 (첫 번째 환자)")
with open(out_path, 'r') as f:
    sample = json.loads(f.readline())

print(f"  subject_id: {sample['subject_id']}")
print(f"  입원 수: {len(sample['admissions'])}")
first_adm = sample['admissions'][0]
print(f"  첫 번째 입원 hadm_id: {first_adm.get('hadm_id')}")
print(f"  첫 번째 입원 admittime: {first_adm.get('admittime')}")
print(f"  진단: {str(first_adm.get('diagnoses_list', ''))[:80]}...")
print(f"  입원 전 omr 기록 수: {len(first_adm.get('pre_admission_omr', []))}")
if first_adm.get('pre_admission_omr'):
    for rec in first_adm['pre_admission_omr'][:5]:
        print(f"    {rec['days_before']}일 전: {rec['name']} = {rec['value']}")
