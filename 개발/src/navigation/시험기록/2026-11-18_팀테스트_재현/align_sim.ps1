# 사용: powershell -File align_sim.ps1 -Mode real   (11-18 실차 잔여 방향 8건)  /  -Mode grid  (지연 · 정지 마찰 · 허용 · 오차 192판)
# Python 없는 PC 에서도 돌릴 수 있게 executor._align 의 전 · 후 로직을 옮긴 것이다. 코드를 고치면 이 파일도 맞춘다.
# _align 이전/이후 로직을 그대로 옮겨 같은 차체 모델(sim_world.SimRobot 회전 부분)로 돌려 본다.
# 차체: 명령 지연 delay_s, 1차 지연 lag_s, 서서 도는 마찰 breakaway_s (sim_world.py:164-184 와 같은 식)
param([string]$Mode = 'grid')

$DT = 1.0 / 16.0
$P = @{ rotate_gain = 2.5; rotate_max_w = 1.6; min_turn_w = 0.8; latency_s = 0.30; coast_s = 0.20; settle_s = 0.30
        turn_moved_rad = 0.035; breakaway_hold_s = 0.60; breakaway_ramp = 1.0; lead_learn = 0.5
        give_up = 3; accept_rad = 0.35 }

function Wrap([double]$a) { while ($a -gt [math]::PI) { $a -= 2 * [math]::PI }; while ($a -lt -[math]::PI) { $a += 2 * [math]::PI }; return $a }
function Clamp([double]$v, [double]$l) { return [math]::Max(-$l, [math]::Min($l, $v)) }
function Sgn([double]$v) { if ($v -ge 0) { return 1.0 } else { return -1.0 } }

function Run-Case([string]$Ver, [double]$Err0Deg, [double]$Tol, [double]$Delay, [double]$Lag, [double]$Break, [double]$Timeout = 30.0) {
    $goal = 0.0
    $yaw = -($Err0Deg * [math]::PI / 180.0)   # err = goal - yaw = Err0
    $t = 0.0; $queue = New-Object System.Collections.ArrayList
    $tw = 0.0; $w = 0.0; $turnSince = $null
    $last = 0.0; $state = 'settle'; $settleSince = $null
    $rotSince = $null; $rotYaw0 = 0.0; $rotErr0 = 0.0; $moved = $false; $lead = $null; $stopRef = $null; $attempts = 0
    $outcome = 'TIMED_OUT'; $detail = ''; $cmds = 0; $stops = 0; $maxAbs = 0.0
    $pulseCount = 0
    while ($t -lt $Timeout) {
        # ---- 제어기 한 틱 (t, yaw) → cmd
        $cmd = 0.0
        $err = Wrap ($goal - $yaw)
        $doAlign = ($state -eq 'align')
        if ($state -eq 'settle') {
            if ($null -eq $settleSince) { $settleSince = $t }
            if (($t - $settleSince) -ge $P.settle_s) {
                $settleSince = $null
                if ([math]::Abs($err) -le $Tol) { $outcome = 'REACHED'; break }
                $state = 'align'; $doAlign = $true
            }
        }
        if ($doAlign) {
            if ($Ver -eq 'old') {
                if ([math]::Abs($err) -le $Tol) { $state = 'settle'; $settleSince = $null; $cmd = 0.0 }
                else {
                    if ([math]::Abs($last) -lt 1e-6 -or $null -eq $rotSince) { $rotSince = $t }
                    $margin = 0.0; if (($t - $rotSince) -ge $P.latency_s) { $margin = [math]::Abs($last) * $P.coast_s }
                    if ([math]::Abs($err) -le $Tol + $margin) { $state = 'settle'; $settleSince = $null; $cmd = 0.0 }
                    else {
                        $cw = Clamp ($P.rotate_gain * $err) $P.rotate_max_w
                        if ([math]::Abs($cw) -lt $P.min_turn_w) { $cw = (Sgn $err) * $P.min_turn_w }
                        $cmd = $cw
                    }
                }
            } else {
                # 학습: 정지 뒤 settle 끝에서 실제로 더 돈 각도
                if ($null -ne $stopRef -and [math]::Abs($last) -le 1e-6) {
                    $extra = [math]::Min([math]::Abs((Wrap ($yaw - $stopRef))), 30.0 * [math]::PI / 180.0); $stopRef = $null
                    if ($null -eq $lead) { $lead = $extra } else { $lead = (1 - $P.lead_learn) * $lead + $P.lead_learn * $extra }
                }
                if ([math]::Abs($err) -le $Tol) { $stopRef = $null; $state = 'settle'; $settleSince = $null; $cmd = 0.0 }
                else {
                    $finished = $false
                    if ([math]::Abs($last) -lt 1e-6 -or $null -eq $rotSince) {
                        if ($P.give_up -gt 0 -and $attempts -ge $P.give_up -and [math]::Abs($err) -le $P.accept_rad) {
                            $outcome = 'REACHED*'; $detail = "허용 밖 $([math]::Round([math]::Abs($err)*180/[math]::PI,1))° 에서 끝냄"; $finished = $true
                        } else {
                            $attempts++; $rotSince = $t; $rotYaw0 = $yaw; $rotErr0 = [math]::Abs($err); $moved = $false
                        }
                    }
                    if ($finished) { break }
                    if (-not $moved -and [math]::Abs((Wrap ($yaw - $rotYaw0))) -ge [math]::Min($P.turn_moved_rad, 0.5 * $rotErr0)) { $moved = $true }
                    $cw = Clamp ($P.rotate_gain * $err) $P.rotate_max_w
                    if ([math]::Abs($cw) -lt $P.min_turn_w) { $cw = (Sgn $err) * $P.min_turn_w }
                    $stopNow = $false
                    if ($moved) {
                        $ld = if ($null -ne $lead) { $lead } else { [math]::Abs($cw) * $P.coast_s }
                        if ([math]::Abs($err) -le $Tol + $ld) { $stopRef = $yaw; $state = 'settle'; $settleSince = $null; $cmd = 0.0; $stopNow = $true }
                    } else {
                        $stuck = $t - $rotSince - $P.breakaway_hold_s
                        if ($stuck -gt 0) { $cw = (Sgn $err) * [math]::Min($P.rotate_max_w, [math]::Abs($cw) + $P.breakaway_ramp * $stuck) }
                    }
                    if (-not $stopNow) { $cmd = $cw }
                }
            }
        }
        if ([math]::Abs($cmd) -gt 1e-6 -and [math]::Abs($last) -le 1e-6) { $pulseCount++ }
        $last = $cmd
        # ---- 차체
        [void]$queue.Add(@(($t + $Delay), $cmd))
        $t += $DT
        $target = $null
        while ($queue.Count -gt 0 -and $queue[0][0] -le $t) { $target = $queue[0]; $queue.RemoveAt(0) }
        if ($null -ne $target) { $tw = $target[1] }
        $twe = $tw
        if ($Break -gt 0 -and [math]::Abs($tw) -gt 1e-6) {
            if ($null -eq $turnSince) { $turnSince = $t }
            if (($t - $turnSince) -lt $Break -and [math]::Abs($w) -lt 0.05) { $twe = 0.0 }
        } else { $turnSince = $null }
        $k = if ($Lag -le 0) { 1.0 } else { [math]::Min(1.0, $DT / $Lag) }
        $w += ($twe - $w) * $k
        $yaw = Wrap ($yaw + $w * $DT)
        $e = [math]::Abs((Wrap ($goal - $yaw))) * 180 / [math]::PI
        if ($e -gt $maxAbs) { $maxAbs = $e }
    }
    $fin = [math]::Abs((Wrap ($goal - $yaw))) * 180 / [math]::PI
    return [pscustomobject]@{ ver = $Ver; err0 = $Err0Deg; tolDeg = [math]::Round($Tol * 180 / [math]::PI, 1); brk = $Break; delay = $Delay
        outcome = $outcome; t = [math]::Round($t, 1); finalErrDeg = [math]::Round($fin, 1); pulses = $pulseCount; detail = $detail }
}

if ($Mode -eq 'real') {
    # 11-18 실차에서 본 남은 방향 · 허용: (오차°, 허용 rad)
    $cases = @(@(10.9, 0.10), @(12.8, 0.10), @(13.4, 0.10), @(15.5, 0.10), @(36.5, 0.5), @(33.1, 0.5), @(28.7, 0.5), @(38.9, 0.5))
    $out = foreach ($c in $cases) { foreach ($ver in 'old', 'new') { Run-Case $ver $c[0] $c[1] 0.30 0.20 0.45 } }
    $out | Format-Table -AutoSize
} else {
    $out = foreach ($brk in 0.0, 0.45, 0.8, 1.2) { foreach ($dly in 0.13, 0.30, 0.50) {
        foreach ($tol in 0.05, 0.10, 0.5) { foreach ($e in 3, 6, 12, 25, 45, 90, 170) {
            if ($e * [math]::PI / 180 -le $tol) { continue }
            foreach ($ver in 'old', 'new') { Run-Case $ver $e $tol $dly 0.20 $brk } } } } }
    $out | Export-Clixml (Join-Path $env:TEMP "align_sim_grid.xml")
    foreach ($ver in 'old', 'new') {
        $s = $out | Where-Object ver -eq $ver
        $strict = $s | Where-Object { $_.outcome -eq 'REACHED' }
        $fallback = $s | Where-Object { $_.outcome -eq 'REACHED*' }
        $to = $s | Where-Object { $_.outcome -eq 'TIMED_OUT' }
        "{0}: 전체 {1}, REACHED(허용 안) {2}, REACHED*(상한으로 끝) {3}, TIMED_OUT {4}, 평균 시간 {5:N1}s (REACHED 계열), 최대 {6:N1}s" -f $ver, $s.Count, $strict.Count, $fallback.Count, $to.Count, (($s | Where-Object { $_.outcome -ne 'TIMED_OUT' } | Measure-Object t -Average).Average), (($s | Measure-Object t -Maximum).Maximum)
    }
}
