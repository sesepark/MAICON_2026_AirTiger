[Console]::OutputEncoding=[Text.Encoding]::UTF8
function Hyp($a,$b){ [math]::Sqrt($a*$a+$b*$b) }
$dir=Join-Path $PSScriptRoot '..\logs'
$rows=[IO.File]::ReadAllLines("$dir\teleop_20261118_150008.jsonl",[Text.Encoding]::UTF8) | % { $_ | ConvertFrom-Json }
$starts=@{};$ends=@{}
$rows|?{$_.ev -eq 'move_start'}|%{$starts[[int]$_.move]=$_}
$rows|?{$_.ev -eq 'move_end'}|%{$ends[[int]$_.move]=$_}
"id|iso|kind|target|lin|ang|drive_s|pub|odom_n|dt_med_s|path_m|fwd_m|dyaw|first_motion_lat_s|motion_after_last_cmd_s|zero_samples_during_drive|samples_during_drive|max_vx|max_wz"
foreach($m in ($starts.Keys|sort)){
 $s=$starts[$m];$e=$ends[$m]
 $od=@($rows|?{$_.ev -eq 'odom' -and $_.move -eq $m}); $cm=@($rows|?{$_.ev -eq 'cmd' -and $_.move -eq $m})
 $t0=$cm[0].t; $tl=$cm[$cm.Count-1].t
 $mov=@(); $path=0; $dt=@()
 for($i=1;$i -lt $od.Count;$i++){ $d=Hyp ($od[$i].x-$od[$i-1].x) ($od[$i].y-$od[$i-1].y); $path+=$d; if($d -gt 1e-6){$mov+=$od[$i]}; $dt+=($od[$i].t-$od[$i-1].t) }
 $lat='n/a';$coast='n/a'
 if($mov.Count -gt 0){ $lat=[math]::Round($mov[0].t-$t0,2); $coast=[math]::Round($mov[$mov.Count-1].t-$tl,2) }
 $dur=@($od|?{$_.t -le $tl}); $vz=@($dur|?{[math]::Abs($_.vx) -lt 1e-6 -and [math]::Abs($_.wz) -lt 1e-6}).Count
 $maxwz=[math]::Round((($od|%{[math]::Abs($_.wz)})|measure -Max).Maximum,2); $maxvx=[math]::Round((($od|%{[math]::Abs($_.vx)})|measure -Max).Maximum,2)
 $dts=@($dt|sort); $med=[math]::Round($dts[[int]($dts.Count/2)],3)
 "{0}|{1}|{2}|{3}|{4}|{5}|{6}|{7}|{8}|{9}|{10:N3}|{11}|{12}|{13}|{14}|{15}|{16}|{17}|{18}" -f $m,$s.iso.Substring(11,8),$s.kind,$s.target,$s.lin,$s.ang,$e.result.cmd.drive_s,$e.published,$od.Count,$med,$path,$e.result.odom_forward_m,$e.result.odom_dyaw_deg,$lat,$coast,$vz,$dur.Count,$maxvx,$maxwz
}
"--- drift between moves (end of N-1 -> start of N)"
foreach($m in ($starts.Keys|sort)){ if($ends.ContainsKey($m-1)){ $pe=$ends[$m-1].odom_end; $cs=$starts[$m].odom_start; "#{0}: dx={1:N4} dy={2:N4}" -f $m,($cs.x-$pe.x),($cs.y-$pe.y) } }
"--- cumulative: start x={0:N3} y={1:N3} end x={2:N3} y={3:N3}" -f $starts[1].odom_start.x,$starts[1].odom_start.y,$ends[16].odom_end.x,$ends[16].odom_end.y
"--- odom stamp vs wall age: first odom per move (stamp gap)"
"--- cmd publish rate: " + (($rows|?{$_.ev -eq 'cmd' -and $_.move -eq 3}|%{$_.since_start}|select -First 4) -join ',')
"--- stop events (cmd_zero why): " + (($rows|?{$_.ev -eq 'cmd_zero'}|group why|%{"$($_.Name)=$($_.Count)"}) -join ', ')
"--- distinct odom x,y,vx values in turn #16 (motion?):"
$t16=@($rows|?{$_.ev -eq 'odom' -and $_.move -eq 16}); "x range {0:N4}..{1:N4}; y range {2:N4}..{3:N4}" -f (($t16|%{$_.x}|measure -Min).Minimum),(($t16|%{$_.x}|measure -Max).Maximum),(($t16|%{$_.y}|measure -Min).Minimum),(($t16|%{$_.y}|measure -Max).Maximum)
"--- moves after #8: aborted #9 (stop button) detail"
$e9=$ends[9].result; "#9 target={0} fwd={1} dist={2} drive_s={3}" -f $e9.target,$e9.odom_forward_m,$e9.odom_dist_m,$e9.cmd.drive_s
