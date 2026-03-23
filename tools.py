"""
워해머 에이지 오브 지그마 4판 전투 기대값 계산 도구

Gemini Function Calling용 함수 모음.
모든 파라미터에 타입 힌트와 docstring을 명시하여
Gemini SDK가 자동으로 스키마를 생성합니다.
"""


def calculate_expected_damage(
    attacks: float,
    to_hit: int,
    to_wound: int,
    rend: int,
    damage: float,
    target_save: int,
    target_ward: int = 0,
) -> str:
    """
    워해머 에이지 오브 지그마 4판 규칙에 따라 1대1 전투의 평균 기대 데미지를 계산합니다.
    사용자가 데미지, 기대값, 얼마나 데미지가 들어가냐, 공격력 계산 등을 물어보면 반드시 이 함수를 호출하세요.

    주사위 규칙:
    - 1은 항상 실패, 6은 항상 성공 (명중/관통/세이브/와드 모두 동일)
    - 렌드는 방어측 세이브 목표값을 높입니다 (예: 세이브 4+, 렌드 1 -> 실질 세이브 5+)
    - 와드는 세이브와 독립적으로 작동합니다

    Args:
        attacks: 평균 공격 횟수. 고정값이면 정수, 주사위면 평균값으로 입력. 예: 3, 3.5 (D6)
        to_hit: 명중 굴림 목표의 숫자. 예: 3은 3+을 의미
        to_wound: 관통 굴림 목표의 숫자. 예: 4는 4+을 의미
        rend: 렌드 수치. 0은 렌드 없음, 1은 렌드 -1 (세이브 1단계 악화)
        damage: 세이브 실패 당 데미지. 고정값이면 정수, 주사위면 평균값으로 입력. 예: 2, 2.0 (D3)
        target_save: 방어측 세이브 목표의 숫자. 예: 4는 4+을 의미
        target_ward: 방어측 와드 세이브 목표의 숫자. 0이면 와드 없음. 예: 6은 6+을 의미

    Returns:
        기대 명중 수, 기대 관통 수, 기대 세이브 실패 수, 최종 기대 데미지를 담은
        상세 결과 텍스트.
    """
    # ── 확률 계산 (1 항상 실패, 6 항상 성공) ──────────────────────────────
    p_hit = max(1 / 6.0, min(5 / 6.0, (7 - to_hit) / 6.0))
    p_wound = max(1 / 6.0, min(5 / 6.0, (7 - to_wound) / 6.0))

    save_effective = target_save + rend
    if save_effective >= 7:
        p_save = 0.0  # 자동 세이브 실패
    else:
        # 1은 항상 세이브 실패이므로 최대 5/6
        p_save = max(0.0, min(5 / 6.0, (7 - save_effective) / 6.0))
    p_fail_save = 1.0 - p_save

    if target_ward > 0:
        p_ward = max(1 / 6.0, min(5 / 6.0, (7 - target_ward) / 6.0))
        p_fail_ward = 1.0 - p_ward
    else:
        p_ward = 0.0
        p_fail_ward = 1.0

    # ── 단계별 기대값 ─────────────────────────────────────────────────────
    expected_hits        = round(attacks * p_hit, 3)
    expected_wounds      = round(expected_hits * p_wound, 3)
    expected_failed_saves = round(expected_wounds * p_fail_save, 3)
    expected_final       = round(expected_failed_saves * p_fail_ward * damage, 3)

    # ── 결과 텍스트 조합 (** 볼드체 사용 금지) ───────────────────────────
    save_eff_str = f"{save_effective}+" if save_effective <= 6 else "자동 실패"
    ward_str = f"{target_ward}+" if target_ward > 0 else "없음"

    lines = [
        "[ 전투 기대 데미지 계산 결과 ]",
        "",
        "[ 입력 스탯 ]",
        f"  공격 횟수    : {attacks}",
        f"  명중 굴림    : {to_hit}+  (성공 확률 {p_hit:.1%})",
        f"  관통 굴림    : {to_wound}+  (성공 확률 {p_wound:.1%})",
        f"  렌드         : {rend}",
        f"  데미지       : {damage}",
        f"  세이브       : {target_save}+  ->  렌드 적용 후 {save_eff_str}  (실패 확률 {p_fail_save:.1%})",
        f"  와드         : {ward_str}" + (f"  (실패 확률 {p_fail_ward:.1%})" if target_ward > 0 else ""),
        "",
        "[ 단계별 기대값 ]",
        f"  기대 명중 수         : {attacks} x {p_hit:.1%} = {expected_hits}",
        f"  기대 관통 수         : {expected_hits} x {p_wound:.1%} = {expected_wounds}",
        f"  기대 세이브 실패 수  : {expected_wounds} x {p_fail_save:.1%} = {expected_failed_saves}",
    ]
    if target_ward > 0:
        lines.append(
            f"  와드 실패 후 기대값  : {expected_failed_saves} x {p_fail_ward:.1%} = "
            f"{round(expected_failed_saves * p_fail_ward, 3)}"
        )
    lines += [
        "",
        f"  최종 기대 데미지 : {expected_failed_saves} x {p_fail_ward:.1%} x {damage} = {expected_final}",
    ]

    return "\n".join(lines)
