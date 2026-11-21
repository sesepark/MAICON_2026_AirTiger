#!/usr/bin/env bash
# 로봇의 ~/team 에 풀어 넣을 묶음(team_bundle.tgz)을 만든다. 저장소가 있는 PC 에서 돌린다 (Windows 는 Git Bash).
#   bash 개발/robot/make_bundle.sh [만들 파일 경로]
# 들어가는 것: team_node.py · programs.json · secrets.example.json · lidar.sh · ws/src/<패키지들> · VERSION(어느 커밋인지)
# 들어가지 않는 것: secrets.json 과 각 프로그램의 설정 파일(supervisor.json 등) — 로봇에 있는 것을 덮어쓰지 않는다
# 로봇에서:  mkdir -p ~/team && tar xzf team_bundle.tgz -C ~/team && cd ~/team/ws && colcon build
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${1:-$repo/team_bundle.tgz}"
tmp="$(mktemp -d)"; chmod 755 "$tmp"  # 묶음의 맨 위 폴더 권한이 700 이면, 풀 때 그 권한이 로봇의 폴더에 옮겨 붙는다
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/ws/src"
cp "$repo/개발/robot/team_node.py" "$repo/개발/robot/programs.json" "$repo/개발/robot/secrets.example.json" "$repo/개발/robot/lidar.sh" "$tmp/"
for pkg in "$repo"/개발/src/*/; do
  if [ -f "$pkg/package.xml" ]; then cp -r "$pkg" "$tmp/ws/src/"; fi
done
find "$tmp" \( -name __pycache__ -o -name '*.egg-info' -o -name .pytest_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
find "$tmp" -name 'secrets.json' -delete
commit="$(git -C "$repo" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
dirty="$(git -C "$repo" status --porcelain -- 개발 2>/dev/null | wc -l | tr -d ' ')"
printf '%s\n' "commit $commit" "uncommitted_files $dirty" "bundled_at $(date '+%Y-%m-%d %H:%M:%S')" > "$tmp/VERSION"
tar czf "$out" -C "$tmp" .
echo "만듦: $out (커밋 $commit, 올리지 않은 수정 파일 $dirty개, $(du -h "$out" | cut -f1))"
