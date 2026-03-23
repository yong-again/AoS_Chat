import json
import os
from pathlib import Path

# JSON 파일들이 저장된 최상위 폴더 경로 (상황에 맞게 수정하세요)
OUTPUT_DIR = "./outputs" 

def validate_faction_db(data, filename):
    errors = []
    if not isinstance(data, dict):
        return [f"최상위 구조가 Dictionary가 아닙니다. (현재: {type(data).__name__})"]
    
    if "aos_matched_play" not in data:
        errors.append("'aos_matched_play' 키가 누락되었습니다.")
    else:
        if "army_rules" not in data["aos_matched_play"]:
            errors.append("aos_matched_play 안에 'army_rules'가 없습니다.")
        if "warscrolls" not in data["aos_matched_play"]:
            errors.append("aos_matched_play 안에 'warscrolls'가 없습니다.")
            
    if "spearhead" not in data:
        errors.append("'spearhead' 키가 누락되었습니다.")
    else:
        if "spearhead_name" not in data["spearhead"]:
            errors.append("spearhead 안에 'spearhead_name'이 없습니다.")
        if "warscrolls" not in data["spearhead"]:
            errors.append("spearhead 안에 'warscrolls'가 없습니다.")
            
    return errors

def validate_balance_db(data, filename):
    errors = []
    if not isinstance(data, list):
        return [f"최상위 구조가 List가 아닙니다. (현재: {type(data).__name__})"]
    
    # 첫 번째 항목의 키값 검사
    if len(data) > 0:
        first_item = data[0]
        expected_keys = {"unit_name", "points", "unit_size", "regiment_options"}
        actual_keys = set(first_item.keys())
        
        if not expected_keys.issubset(actual_keys):
            missing = expected_keys - actual_keys
            errors.append(f"표준 스키마 키 누락 (또는 임의 개명): {missing}")
            
    return errors

def validate_spearhead_db(data, filename):
    errors = []
    if not isinstance(data, dict):
        return [f"최상위 구조가 Dictionary가 아닙니다. (현재: {type(data).__name__})"]
    
    if "spearhead" not in data:
        errors.append("'spearhead' 키가 누락되었습니다. (단독 스피어헤드 문서가 아닐 수 있음)")
    else:
        if "spearhead_name" not in data["spearhead"]:
            errors.append("spearhead 안에 'spearhead_name'이 없습니다.")
        if "warscrolls" not in data["spearhead"]:
            errors.append("spearhead 안에 'warscrolls'가 없습니다.")
            
    return errors

def run_validation():
    print("=== 🔍 JSON 스키마 검증 시작 ===")
    
    base_path = Path(OUTPUT_DIR)
    if not base_path.exists():
        print(f"경로를 찾을 수 없습니다: {base_path}")
        return

    total_files = 0
    error_files = 0
    
    # 모든 json 파일 순회
    for filepath in base_path.rglob("*.json"):
        total_files += 1
        filename = filepath.name
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"\n❌ [파싱 에러] {filename}\n   - JSON 구조가 깨졌습니다: {e}")
            error_files += 1
            continue

        errors = []
        
        # 파일명 접두사에 따른 검사 분기
        if filename.startswith("faction_db"):
            errors = validate_faction_db(data, filename)
        elif filename.startswith("balance_db"):
            errors = validate_balance_db(data, filename)
        elif filename.startswith("spearhead_db"):
            # 기존 팩션 팩에서 분리된 스피어헤드인지, 단독 스피어헤드 문서인지에 따라 검사
            errors = validate_spearhead_db(data, filename)
        elif filename.startswith("other_db"):
            if not isinstance(data, list):
                errors.append(f"최상위 구조가 List가 아닙니다. (현재: {type(data).__name__})")
        # rule_db는 상대적으로 구조가 유연하므로 심각한 에러만 아니면 패스
        
        if errors:
            print(f"\n⚠️ [스키마 불일치] {filename}")
            for err in errors:
                print(f"   - {err}")
            error_files += 1

    print(f"\n=== ✅ 검증 완료 ===")
    print(f"총 검사한 파일: {total_files}개")
    print(f"문제가 발견된 파일: {error_files}개")

if __name__ == "__main__":
    run_validation()