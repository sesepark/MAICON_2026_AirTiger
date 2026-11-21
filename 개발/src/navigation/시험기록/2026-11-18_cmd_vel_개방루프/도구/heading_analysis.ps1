[Console]::OutputEncoding=[Text.Encoding]::UTF8
$rows=[IO.File]::ReadAllLines((Join-Path $PSScriptRoot '..\logs\teleop_20261118_150008.jsonl'),[Text.Encoding]::UTF8) | % { $_ | ConvertFrom-Json }
$ends=@{};$starts=@{}
$rows|?{$_.ev -eq 'move_end'}|%{$ends[[int]$_.move]=$_}
$rows|?{$_.ev -eq 'move_start'}|%{$starts[[int]$_.move]=$_}
"move | kind target | dx dy (m) | atan2(dy,dx) deg | dir of cmd | implied internal yaw (deg, mod 360)"
foreach($m in ($starts.Keys|sort)){
 $a=$starts[$m].odom_start;$b=$ends[$m].odom_end
 $dx=$b.x-$a.x;$dy=$b.y-$a.y
 $ang=[math]::Atan2($dy,$dx)*180/[math]::PI
 $kind=$starts[$m].kind;$tgt=$starts[$m].target
 $impl='-'
 if($kind -eq 'dist' -and ([math]::Sqrt($dx*$dx+$dy*$dy) -gt 0.03)){ $y=$ang; if($tgt -lt 0){$y=$ang-180}; $impl=[math]::Round((($y%360)+360)%360,0) }
 "{0,2} | {1} {2,6} | {3,7:N4} {4,7:N4} | {5,7:N1} | {6} | {7}" -f $m,$kind,$tgt,$dx,$dy,$ang,$(if($tgt -lt 0){'back/right'}else{'fwd/left'}),$impl
}
