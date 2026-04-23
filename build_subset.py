import pandas as pd
import json
from pathlib import Path

PROCESSED = Path('./processed/')
SUBSET    = Path('./processed/subset/')
SUBSET.mkdir(exist_ok=True)

TARGET = 1000

print("=" * 60)
print("풍부한 환자 서브셋 선택")
print("=" * 60)

# ── 1. 기본 파일 로드 ─────────────────────────────────────────
print("\n▶ 파일 로드 중...")
adm = pd.read_csv(PROCESSED / 'admission_profile.csv')
pat = pd.read_csv(PROCESSED / 'patient_summary.csv')
print(f"  admission_profile: {len(adm):,}행")
print(f"  patient_summary:   {len(pat):,}행")

# ── 2. 입원 단위 필터: 검사 10건 이상인 입원만 ────────────────
rich_adm = adm[adm['n_labs'] >= 10]

# subject_id별 집계 (n_admissions는 patient_summary에 이미 있으므로 제외)
adm_stats = adm.groupby('subject_id').agg(
    n_rich_adm      = ('n_labs',          lambda x: (x >= 10).sum()),
    max_labs        = ('n_labs',          'max'),
    max_diagnoses   = ('n_diagnoses',     'max'),
    has_icu         = ('has_icu',         'any'),
    total_abnormal  = ('n_abnormal_labs', 'sum'),
    total_rx        = ('n_prescriptions', 'sum'),
).reset_index()

# patient_summary와 합치기
merged = pat.merge(adm_stats, on='subject_id', how='inner')

# ── 3. 풍부도 점수 계산 ───────────────────────────────────────
merged['score'] = (
    merged['n_admissions']    * 2  +
    merged['n_rich_adm']      * 3  +
    merged['max_labs']        * 0.1 +
    merged['max_diagnoses']   * 1  +
    merged['has_icu'].astype(int) * 5 +
    merged['total_abnormal']  * 0.05 +
    merged['n_observations'].fillna(0) * 0.1 +
    merged['n_hospital_deaths'].fillna(0) * 3
)

# ── 4. 그룹별 선택 (다양성 보장) ─────────────────────────────
print("\n▶ 그룹별 선택 중...")

def pick(df, n, label):
    selected = df.nlargest(n, 'score')['subject_id'].tolist()
    print(f"  {label}: {len(selected)}명")
    return set(selected)

# 그룹 1: 다입원 + 풍부한 검사
g1_pool = merged[
    (merged['n_admissions'] >= 2) &
    (merged['n_rich_adm']   >= 2) &
    (merged['max_labs']     >= 20)
]
g1 = pick(g1_pool, 400, "다입원 + 풍부한 검사")

# 그룹 2: ICU 경험
g2_pool = merged[
    merged['has_icu'] &
    (merged['max_labs'] >= 10) &
    ~merged['subject_id'].isin(g1)
]
g2 = pick(g2_pool, 300, "ICU 경험")

# 그룹 3: 원내 사망
g3_pool = merged[
    (merged['n_hospital_deaths'] >= 1) &
    ~merged['subject_id'].isin(g1 | g2)
]
g3 = pick(g3_pool, 150, "원내 사망")

# 그룹 4: 외래 이력 풍부 (omr)
g4_pool = merged[
    (merged['n_observations'].fillna(0) >= 10) &
    (merged['n_admissions'] >= 2) &
    ~merged['subject_id'].isin(g1 | g2 | g3)
]
g4 = pick(g4_pool, 150, "외래 이력 풍부")

selected_ids = g1 | g2 | g3 | g4
print(f"\n  총 선택: {len(selected_ids):,}명")

# ── 5. subset 파일 저장 ───────────────────────────────────────
print("\n▶ 서브셋 파일 저장 중...")

# patient_summary 서브셋
subset_pat = pat[pat['subject_id'].isin(selected_ids)]
subset_pat.to_csv(SUBSET / 'patient_summary.csv', index=False)
print(f"  patient_summary.csv: {len(subset_pat):,}명")

# admission_profile 서브셋
subset_adm = adm[adm['subject_id'].isin(selected_ids)]
subset_adm.to_csv(SUBSET / 'admission_profile.csv', index=False)
print(f"  admission_profile.csv: {len(subset_adm):,}건 입원")

# patient_dict.jsonl 서브셋
print("  patient_dict.jsonl 필터링 중...")
n_written = 0
with open(PROCESSED / 'patient_dict.jsonl', 'r') as fin, \
     open(SUBSET    / 'patient_dict.jsonl', 'w') as fout:
    for line in fin:
        record = json.loads(line)
        if record['subject_id'] in selected_ids:
            fout.write(line)
            n_written += 1
print(f"  patient_dict.jsonl: {n_written:,}명")

# ── 6. 서브셋 요약 ────────────────────────────────────────────
print("\n" + "=" * 60)
print("서브셋 완료!")
print(f"  환자 수:        {len(subset_pat):,}명")
print(f"  입원 건수:      {len(subset_adm):,}건")
print(f"  ICU 경험:       {subset_pat['total_icu_stays'].gt(0).sum():,}명")
print(f"  원내 사망:      {subset_pat['n_hospital_deaths'].gt(0).sum():,}명")
print(f"  omr 기록 있음:  {subset_pat['n_observations'].gt(0).sum():,}명")
print(f"  평균 입원 횟수: {subset_adm.groupby('subject_id').size().mean():.1f}회")
print()

import os
for f in SUBSET.iterdir():
    size_mb = os.path.getsize(f) / 1024 / 1024
    print(f"  {f.name}: {size_mb:.1f} MB")
print("=" * 60)
