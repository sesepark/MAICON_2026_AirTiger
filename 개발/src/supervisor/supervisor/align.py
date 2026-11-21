"""사진 한 장의 그림들을 지도의 면들에 「한꺼번에」 맞춘다 (ROS 없이 시험할 수 있는 계산만).

왜 (2026-11-19 14:52 실차의 사진 108장을 정답표와 견준 결과):
  판독 모델은 2 ~ 3.5 m 떨어진 그림도 맞게 읽는다. 멀리서 틀리게 보인 것은 거의 다 「그림은 맞게 읽었는데 옆 면의 것으로 붙인」 경우였다
  (붙어 있는 두 상자의 면은 2 m 에서 사진 폭의 0.09, 2.5 m 에서 0.07 떨어져 찍힌다. 차체 방향이 5° 틀리면 사진이 통째로 0.04 밀린다).
  그림을 하나씩 따로 붙이면(pick_each) 이 밀림을 알 수 없어, 헷갈릴 만한 것은 모두 버려야 했다 — 그래서 1.6 m 안의 면만 적었다.

여기서는: 사진의 그림 여럿과 면 여럿이 「같은 만큼 밀려」 맞는 자리를 찾는다. 밀림을 뺀 뒤에는 그림과 면의 자리가 0.025 안으로 맞아야 하고,
그렇게 맞는 짝이 둘 이상이어야 하며, 같은 수의 짝이 나오는 다른 밀림이 없어야 한다 (한 칸 밀려 맞는 경우를 거른다).
"""
from __future__ import annotations

from . import arena as A


def _sized(q):
    return len(q) >= 5 and q[3] and q[4]


def aligned(expected, pictures, max_shift=0.07, tol=0.025, heights=(0.7, 1.6), widths=(0.45, 1.9), min_pairs=2, min_conf=0.5,
            min_fraction=0.7, max_dist=4.0, max_incidence_deg=72.0):
    """사진의 그림들을 면들에 한꺼번에 맞춘다. 돌려주는 값: (밀림, [(Expected, kind, 자리 차이, 그림)]) — 못 맞추면 (None, []).

    expected = faces.expected_faces 의 답. pictures = [(kind, x, cut, 가로, 세로, 확신)].
    쓰는 그림: 크기가 있고, 가장자리에서 잘리지 않았고, 「모름」 이 아니고, 확신이 min_conf 이상인 것.
    쓰는 면: 보이는 비율 min_fraction 이상 · max_dist 안 · max_incidence_deg 안 (맞추는 데 쓰는 한도다 — 적을지는 부르는 쪽이 더 좁게 정한다).
    """
    faces = [e for e in expected if e.fraction >= min_fraction and e.dist <= max_dist and e.incidence_deg <= max_incidence_deg
             and e.width > 0.0 and e.height > 0.0]
    pics = [q for q in pictures or () if _sized(q) and not q[2] and q[0] != A.UNKNOWN and q[0] in A.ALL_KINDS
            and (len(q) < 6 or q[5] is None or q[5] >= min_conf)]
    if len(faces) < min_pairs or len(pics) < min_pairs:
        return None, []

    def fits(q, e):
        return heights[0] <= q[4] / e.height <= heights[1] and widths[0] <= q[3] / e.width <= widths[1]

    def pairs_at(shift):
        """그 밀림에서 서로 가장 가까운 그림 · 면의 짝들 (자리 차이 tol 안 · 크기가 맞는 것)."""
        cand = sorted(((abs(q[1] - e.x - shift), i, j) for i, q in enumerate(pics) for j, e in enumerate(faces)
                       if abs(q[1] - e.x - shift) <= tol and fits(q, e)), key=lambda c: c[0])
        used_p, used_f, out = set(), set(), []
        for d, i, j in cand:
            if i in used_p or j in used_f:
                continue
            used_p.add(i)
            used_f.add(j)
            out.append((i, j, d))
        return out

    shifts = sorted({round(q[1] - e.x, 4) for q in pics for e in faces if abs(q[1] - e.x) <= max_shift and fits(q, e)})
    best = None
    seen = []
    for s in shifts:
        got = pairs_at(s)
        if len(got) < min_pairs:
            continue
        mean = sum(pics[i][1] - faces[j].x for i, j, _d in got) / len(got)  # 그 짝들의 밀림의 평균으로 다시
        got = pairs_at(mean)
        if len(got) < min_pairs:
            continue
        key = frozenset((i, j) for i, j, _d in got)
        seen.append((len(got), sum(d for _i, _j, d in got), mean, key, got))
        if best is None or (len(got), -seen[-1][1]) > (best[0], -best[1]):
            best = seen[-1]
    if best is None:
        return None, []
    # 같은 수(또는 하나 적은 수)의 짝이 나오는 「다른 짝짓기」 가 있으면 어느 쪽인지 알 수 없다
    for n, _resid, _mean, key, _got in seen:
        if key != best[3] and n >= best[0] - (1 if best[0] >= 4 else 0) and not key <= best[3]:
            return None, []
    return best[2], [(faces[j], pics[i][0], d, pics[i]) for i, j, d in best[4]]
