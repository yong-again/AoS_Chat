"""
워해머 에이지 오브 지그마 전투 수학 연산 엔진 (Mathhammer Engine)

지원 기능:
  - AoS 스탯 파서: '3+' → 확률, 'D6' → 평균, '2D3+1' → 평균 등
  - 기대값(Expected Value) 공식 계산
  - 몬테카를로 시뮬레이션 (기본 10,000회)

AoS 4th Edition 규칙 준수:
  - Crit (Mortal): 명중 6 → 1 추가 모탈 데미지 (공격은 계속 진행)
  - Anti-charge (+1 Rend): 대상이 이번 턴 돌격했을 때 렌드 +1
  - All-out Attack: 명중 굴림 +1 (hit_target -1)
  - All-out Defence: 세이브 굴림 +1 (save_target -1)
  - Ward: 세이브 실패 후 와드 굴림 성공 시 피해 무효
"""
import re
import random
from typing import Optional


# ── 스탯 파서 ──────────────────────────────────────────────────────────────────

def parse_roll_target(s: str) -> float:
    """
    '3+' 형식의 굴림 목표값을 성공 확률로 변환합니다.
    '3+' → (7-3)/6 = 4/6 ≈ 0.667
    """
    m = re.match(r"^(\d+)\+$", str(s).strip())
    if m:
        n = int(m.group(1))
        return max(0.0, min(1.0, (7 - n) / 6.0))
    return 0.0


def parse_dice_avg(s: str) -> float:
    """
    주사위 표현식을 평균값으로 변환합니다.
    'D3' → 2.0, 'D6' → 3.5, '2D6' → 7.0, '2D3+1' → 5.0, '3' → 3.0
    """
    s = str(s).strip().upper()
    if s in ("-", "", "NONE", "N/A", "–"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        pass
    m = re.match(r"^(\d*)D(\d+)([+-]\d+)?$", s)
    if m:
        count = int(m.group(1)) if m.group(1) else 1
        sides = int(m.group(2))
        mod   = int(m.group(3)) if m.group(3) else 0
        return count * (sides + 1) / 2.0 + mod
    return 0.0


def _roll_dice(s: str) -> int:
    """주사위 표현식을 실제로 굴립니다. 몬테카를로 내부용."""
    s = str(s).strip().upper()
    if s in ("-", "", "NONE", "N/A", "–"):
        return 0
    try:
        return max(0, int(float(s)))
    except ValueError:
        pass
    m = re.match(r"^(\d*)D(\d+)([+-]\d+)?$", s)
    if m:
        count = int(m.group(1)) if m.group(1) else 1
        sides = int(m.group(2))
        mod   = int(m.group(3)) if m.group(3) else 0
        return max(0, sum(random.randint(1, sides) for _ in range(count)) + mod)
    return 0


def parse_rend(s) -> int:
    """렌드 값을 정수로 변환합니다. '-' → 0, '1' → 1, '2' → 2"""
    if s in (None, "-", "", "N/A", "–", "none"):
        return 0
    try:
        return abs(int(str(s).strip()))
    except (ValueError, TypeError):
        return 0


def _fail_save_prob(save: str, rend: int) -> float:
    """
    세이브 실패 확률을 계산합니다.
    save='3+', rend=1 → effective=4+ → P(fail) = 3/6 = 0.5
    effective >= 7 이면 자동 실패 → P(fail) = 1.0
    """
    m = re.match(r"^(\d+)\+$", str(save).strip())
    if not m:
        return 1.0
    base = int(m.group(1))
    effective = base + rend
    p_save = max(0.0, (7 - effective) / 6.0)
    return 1.0 - p_save


# ── 퍼블릭 API ─────────────────────────────────────────────────────────────────

def calculate_combat_damage(
    attacks:           str,
    hit:               str,
    wound:             str,
    rend:              str,
    damage:            str,
    attacker_count:    int  = 1,
    target_save:       str  = "4+",
    target_ward:       Optional[str] = None,
    all_out_attack:    bool = False,
    all_out_defence:   bool = False,
    charged:           bool = False,
    crit_mortal:       bool = False,
    anti_charge_rend:  bool = False,
    use_monte_carlo:   bool = False,
    iterations:        int  = 10000,
) -> dict:
    """
    워해머 AoS 전투 기대 데미지를 계산합니다.

    Parameters
    ----------
    attacks          : 공격 횟수 ('2', 'D6', '2D3')
    hit              : 명중 굴림 목표 ('3+', '4+')
    wound            : 관통 굴림 목표 ('3+', '4+')
    rend             : 렌드 ('-', '1', '2')
    damage           : 데미지 ('1', 'D3', '2')
    attacker_count   : 공격하는 모델 수
    target_save      : 방어측 기본 세이브 ('3+', '4+', '5+')
    target_ward      : 방어측 와드 세이브 (없으면 None, 예: '6+')
    all_out_attack   : All-out Attack 지휘 능력 사용 여부 (명중 +1)
    all_out_defence  : All-out Defence 지휘 능력 사용 여부 (세이브 +1)
    charged          : 공격 유닛이 이번 턴 돌격했는지 (Anti-charge 트리거)
    crit_mortal      : Crit (Mortal) 무기 특성 보유 여부
    anti_charge_rend : Anti-charge (+1 Rend) 무기 특성 보유 여부
    use_monte_carlo  : True면 몬테카를로, False면 EV 공식 사용
    iterations       : 몬테카를로 반복 횟수 (기본 10,000)
    """
    if use_monte_carlo:
        return _monte_carlo(
            attacks, hit, wound, rend, damage, attacker_count,
            target_save, target_ward, all_out_attack, all_out_defence,
            charged, crit_mortal, anti_charge_rend, iterations,
        )
    return _expected_value(
        attacks, hit, wound, rend, damage, attacker_count,
        target_save, target_ward, all_out_attack, all_out_defence,
        charged, crit_mortal, anti_charge_rend,
    )


# ── 내부 구현 ──────────────────────────────────────────────────────────────────

def _expected_value(
    attacks, hit, wound, rend, damage,
    attacker_count, target_save, target_ward,
    all_out_attack, all_out_defence, charged, crit_mortal, anti_charge_rend,
) -> dict:
    rend_val = parse_rend(rend)
    if anti_charge_rend and charged:
        rend_val += 1

    p_hit = parse_roll_target(hit)
    if all_out_attack:
        p_hit = min(1.0, p_hit + 1 / 6.0)   # 명중 목표 1단계 향상 = +1/6

    p_wound = parse_roll_target(wound)

    effective_rend = max(0, rend_val - (1 if all_out_defence else 0))
    p_fail_save = _fail_save_prob(target_save, effective_rend)

    p_no_ward = 1.0
    if target_ward:
        p_no_ward = 1.0 - parse_roll_target(target_ward)

    n_attacks = parse_dice_avg(attacks) * attacker_count
    avg_dmg   = parse_dice_avg(damage)

    # Crit (Mortal): 6 명중 → 모탈 1 (세이브/와드 무시), 공격도 계속 진행
    p_crit   = 1 / 6.0 if crit_mortal else 0.0
    crit_ev  = n_attacks * p_crit * 1.0          # 1 mortal dmg per crit
    normal_ev = n_attacks * p_hit * p_wound * p_fail_save * p_no_ward * avg_dmg
    total_ev  = round(normal_ev + crit_ev, 3)

    return {
        "method": "expected_value",
        "expected_damage": total_ev,
        "breakdown": {
            "total_attacks":               round(n_attacks, 2),
            "p_hit":                       round(p_hit, 3),
            "p_wound":                     round(p_wound, 3),
            "effective_rend":              effective_rend,
            "p_fail_save":                 round(p_fail_save, 3),
            "p_no_ward":                   round(p_no_ward, 3),
            "avg_damage_per_unsaved_wound": round(avg_dmg, 2),
            "crit_mortal_ev":              round(crit_ev, 3),
            "normal_ev":                   round(normal_ev, 3),
        },
        "modifiers_applied": {
            "all_out_attack":              all_out_attack,
            "all_out_defence":             all_out_defence,
            "anti_charge_rend_triggered":  anti_charge_rend and charged,
        },
    }


def _monte_carlo(
    attacks, hit, wound, rend, damage,
    attacker_count, target_save, target_ward,
    all_out_attack, all_out_defence, charged, crit_mortal, anti_charge_rend,
    iterations,
) -> dict:
    rend_val = parse_rend(rend)
    if anti_charge_rend and charged:
        rend_val += 1

    hit_m = re.match(r"^(\d+)\+$", str(hit).strip())
    hit_target = int(hit_m.group(1)) if hit_m else 4
    if all_out_attack:
        hit_target = max(2, hit_target - 1)

    wound_m = re.match(r"^(\d+)\+$", str(wound).strip())
    wound_target = int(wound_m.group(1)) if wound_m else 4

    save_m = re.match(r"^(\d+)\+$", str(target_save).strip())
    save_base = int(save_m.group(1)) if save_m else 7
    effective_rend = max(0, rend_val - (1 if all_out_defence else 0))
    save_effective = save_base + effective_rend  # 이 값 미만이면 세이브 실패

    ward_target = None
    if target_ward:
        ward_m = re.match(r"^(\d+)\+$", str(target_ward).strip())
        if ward_m:
            ward_target = int(ward_m.group(1))

    results = []
    for _ in range(iterations):
        n_attacks = _roll_dice(attacks) * attacker_count
        total = 0
        for _ in range(int(n_attacks)):
            hit_roll = random.randint(1, 6)
            is_crit  = (hit_roll == 6)

            # Crit (Mortal): 모탈 1 추가 (세이브/와드 무시), 이후 공격 계속
            if crit_mortal and is_crit:
                total += 1

            # 명중 실패
            if hit_roll < hit_target:
                continue

            # 관통 실패
            if random.randint(1, 6) < wound_target:
                continue

            # 세이브 (save_effective >= 7 이면 자동 실패)
            if save_effective < 7 and random.randint(1, 6) >= save_effective:
                continue  # 세이브 성공

            # 와드 세이브
            if ward_target and random.randint(1, 6) >= ward_target:
                continue  # 와드 성공

            total += _roll_dice(damage)

        results.append(total)

    results.sort()
    avg      = sum(results) / iterations
    p10      = results[max(0, int(iterations * 0.10))]
    p50      = results[int(iterations * 0.50)]
    p90      = results[min(iterations - 1, int(iterations * 0.90))]
    zero_pct = round(results.count(0) / iterations * 100, 1)

    return {
        "method":           "monte_carlo",
        "iterations":       iterations,
        "expected_damage":  round(avg, 3),
        "percentiles": {
            "10th_percentile": p10,
            "median":          p50,
            "90th_percentile": p90,
        },
        "prob_zero_damage":  f"{zero_pct}%",
        "modifiers_applied": {
            "all_out_attack":             all_out_attack,
            "all_out_defence":            all_out_defence,
            "anti_charge_rend_triggered": anti_charge_rend and charged,
        },
    }
