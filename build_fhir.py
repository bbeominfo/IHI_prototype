import json
import uuid
import pandas as pd
from pathlib import Path

SUBSET     = Path('./processed/subset/')
FHIR_OUT   = Path('./fhir_output/')
FHIR_OUT.mkdir(exist_ok=True)

# LOINC 코드 매핑 (omr result_name → LOINC)
LOINC = {
    'Blood Pressure':              ('55284-4', 'Blood pressure panel'),
    'Blood Pressure Sitting':      ('55284-4', 'Blood pressure panel'),
    'Blood Pressure Standing':     ('55284-4', 'Blood pressure panel'),
    'Blood Pressure Lying':        ('55284-4', 'Blood pressure panel'),
    'BMI (kg/m2)':                 ('39156-5', 'Body mass index (BMI)'),
    'BMI':                         ('39156-5', 'Body mass index (BMI)'),
    'Weight (Lbs)':                ('29463-7', 'Body weight'),
    'Weight':                      ('29463-7', 'Body weight'),
    'Height (Inches)':             ('8302-2',  'Body height'),
    'Height':                      ('8302-2',  'Body height'),
    'eGFR':                        ('33914-3', 'Glomerular filtration rate'),
}

def icd_system(code: str) -> str:
    return (
        'http://hl7.org/fhir/sid/icd-10-cm'
        if code and code[0].isalpha()
        else 'http://hl7.org/fhir/sid/icd-9-cm'
    )

def entry(full_url: str, resource: dict, method='POST') -> dict:
    return {
        'fullUrl': full_url,
        'resource': resource,
        'request': {'method': method, 'url': resource['resourceType']},
    }

def make_patient(info: dict, subject_id: int, pt_uuid: str) -> dict:
    gender_map = {'M': 'male', 'F': 'female'}
    resource = {
        'resourceType': 'Patient',
        'id': pt_uuid,
        'identifier': [{'system': 'urn:mimic:subject_id', 'value': str(subject_id)}],
        'gender': gender_map.get(info.get('gender', ''), 'unknown'),
    }
    # 출생연도 추정 (anchor_year - anchor_age)
    if info.get('anchor_year') and info.get('anchor_age'):
        birth_year = int(info['anchor_year']) - int(info['anchor_age'])
        resource['birthDate'] = str(birth_year)
    # 사망일
    if info.get('dod') and str(info['dod']) not in ('', 'nan', 'NaT'):
        resource['deceasedDateTime'] = str(info['dod'])
    return resource

def make_encounter(adm: dict, enc_uuid: str, pt_uuid: str) -> dict:
    admission_class = {
        'EW EMER.':                  {'code': 'EMER',  'display': 'Emergency'},
        'URGENT':                    {'code': 'IMP',   'display': 'Inpatient'},
        'DIRECT EMER.':              {'code': 'EMER',  'display': 'Emergency'},
        'OBSERVATION ADMIT':         {'code': 'OBSENC','display': 'Observation Encounter'},
        'SURGICAL SAME DAY ADMISSION':{'code': 'AMB',  'display': 'Ambulatory'},
        'AMBULATORY OBSERVATION':    {'code': 'AMB',   'display': 'Ambulatory'},
        'DIRECT OBSERVATION':        {'code': 'OBSENC','display': 'Observation Encounter'},
        'EU OBSERVATION':            {'code': 'OBSENC','display': 'Observation Encounter'},
    }
    adm_type  = str(adm.get('admission_type', ''))
    cls_info  = admission_class.get(adm_type, {'code': 'IMP', 'display': 'Inpatient'})
    expired   = adm.get('hospital_expire_flag', 0)

    resource = {
        'resourceType': 'Encounter',
        'id': enc_uuid,
        'identifier': [{'system': 'urn:mimic:hadm_id', 'value': str(adm.get('hadm_id', ''))}],
        'status': 'finished',
        'class': {
            'system':  'http://terminology.hl7.org/CodeSystem/v3-ActCode',
            'code':    cls_info['code'],
            'display': cls_info['display'],
        },
        'type': [{'text': adm_type}],
        'subject': {'reference': f'urn:uuid:{pt_uuid}'},
        'period': {
            'start': str(adm.get('admittime', '')),
            'end':   str(adm.get('dischtime', '')),
        },
    }
    if expired:
        resource['hospitalization'] = {
            'dischargeDisposition': {
                'coding': [{'code': 'exp', 'display': 'Expired'}]
            }
        }
    return resource

def make_conditions(adm: dict, enc_uuid: str, pt_uuid: str) -> list:
    codes  = [c.strip() for c in str(adm.get('icd_codes_list',  '')).split('|') if c.strip()]
    names  = [n.strip() for n in str(adm.get('diagnoses_list', '')).split('|') if n.strip()]
    resources = []
    for i, code in enumerate(codes):
        name = names[i] if i < len(names) else code
        resources.append({
            'resourceType': 'Condition',
            'id': str(uuid.uuid4()),
            'clinicalStatus': {
                'coding': [{'system': 'http://terminology.hl7.org/CodeSystem/condition-clinical',
                            'code': 'active'}]
            },
            'code': {
                'coding': [{'system': icd_system(code), 'code': code, 'display': name}],
                'text': name,
            },
            'subject':   {'reference': f'urn:uuid:{pt_uuid}'},
            'encounter': {'reference': f'urn:uuid:{enc_uuid}'},
        })
    return resources

def make_observations(adm: dict, enc_uuid: str, pt_uuid: str) -> list:
    pre_omr   = adm.get('pre_admission_omr', [])
    admittime = pd.to_datetime(adm.get('admittime', ''))
    resources = []
    for rec in pre_omr:
        name       = rec.get('name', '')
        value      = rec.get('value', '')
        days_before= rec.get('days_before', 0)
        loinc_info = LOINC.get(name)
        effective  = (admittime - pd.Timedelta(days=days_before)).strftime('%Y-%m-%dT%H:%M:%S')

        obs = {
            'resourceType': 'Observation',
            'id': str(uuid.uuid4()),
            'status': 'final',
            'subject':   {'reference': f'urn:uuid:{pt_uuid}'},
            'encounter': {'reference': f'urn:uuid:{enc_uuid}'},
            'effectiveDateTime': effective,
            'valueString': str(value),
        }
        if loinc_info:
            obs['code'] = {
                'coding': [{'system': 'http://loinc.org',
                            'code': loinc_info[0], 'display': loinc_info[1]}],
                'text': name,
            }
        else:
            obs['code'] = {'text': name}
        resources.append(obs)
    return resources

def make_medications(adm: dict, enc_uuid: str, pt_uuid: str) -> list:
    drugs = [d.strip() for d in str(adm.get('drugs_list', '')).split('|') if d.strip()]
    resources = []
    for drug in drugs:
        resources.append({
            'resourceType': 'MedicationRequest',
            'id': str(uuid.uuid4()),
            'status': 'completed',
            'intent': 'order',
            'medicationCodeableConcept': {'text': drug},
            'subject':   {'reference': f'urn:uuid:{pt_uuid}'},
            'encounter': {'reference': f'urn:uuid:{enc_uuid}'},
        })
    return resources

def make_procedures(adm: dict, enc_uuid: str, pt_uuid: str) -> list:
    labels = [p.strip() for p in str(adm.get('procedure_labels', '')).split('|') if p.strip()]
    resources = []
    for label in labels:
        resources.append({
            'resourceType': 'Procedure',
            'id': str(uuid.uuid4()),
            'status': 'completed',
            'code': {'text': label},
            'subject':   {'reference': f'urn:uuid:{pt_uuid}'},
            'encounter': {'reference': f'urn:uuid:{enc_uuid}'},
            'performedDateTime': str(adm.get('admittime', '')),
        })
    return resources

def patient_to_bundle(record: dict) -> dict:
    subject_id = record['subject_id']
    info       = record.get('info', {})
    admissions = record.get('admissions', [])

    pt_uuid = str(uuid.uuid4())
    entries = [entry(f'urn:uuid:{pt_uuid}', make_patient(info, subject_id, pt_uuid))]

    for adm in admissions:
        enc_uuid = str(uuid.uuid4())
        entries.append(entry(f'urn:uuid:{enc_uuid}', make_encounter(adm, enc_uuid, pt_uuid)))

        for res in make_conditions(adm, enc_uuid, pt_uuid):
            entries.append(entry(f'urn:uuid:{res["id"]}', res))
        for res in make_observations(adm, enc_uuid, pt_uuid):
            entries.append(entry(f'urn:uuid:{res["id"]}', res))
        for res in make_medications(adm, enc_uuid, pt_uuid):
            entries.append(entry(f'urn:uuid:{res["id"]}', res))
        for res in make_procedures(adm, enc_uuid, pt_uuid):
            entries.append(entry(f'urn:uuid:{res["id"]}', res))

    return {'resourceType': 'Bundle', 'type': 'transaction', 'entry': entries}

# ── 실행 ──────────────────────────────────────────────────────
print("=" * 60)
print("MIMIC-IV → FHIR Transaction Bundle 변환")
print("=" * 60)

n_patients = 0
n_resources = 0

with open(SUBSET / 'patient_dict.jsonl', 'r') as f:
    for line in f:
        record = json.loads(line)
        bundle = patient_to_bundle(record)

        out_path = FHIR_OUT / f'bundle_{record["subject_id"]}.json'
        with open(out_path, 'w', encoding='utf-8') as out:
            json.dump(bundle, out, ensure_ascii=False, indent=2)

        n_patients  += 1
        n_resources += len(bundle['entry'])

        if n_patients % 100 == 0:
            print(f"  {n_patients}명 변환 완료...")

print(f"\n완료!")
print(f"  변환 환자 수:     {n_patients:,}명")
print(f"  총 FHIR 리소스:   {n_resources:,}개")
print(f"  평균 리소스/환자: {n_resources/n_patients:.0f}개")
print(f"  저장 위치:        fhir_output/ ({n_patients}개 파일)")
