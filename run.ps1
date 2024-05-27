Get-ChildItem images/input -Filter *.jpg | 
Foreach-Object {
  $image = $_.FullName
  $output = "images\output\edge_result$($_.BaseName).jpg"
  Write-Host "Running $image" -ForegroundColor Blue
  for ($i = 1; $i -le 6; $i++) {
    $avg = 0
    for ($j = 1; $j -le 10; $j++) {
      $time = Measure-Command { mpiexec -np $i python .\main.py $image $output }
      $avg += $time.TotalMilliseconds
    }
    Write-Host "$i Processes: " -NoNewline
    Write-Host "$($avg/10) ms" -ForegroundColor Red
  }
  Write-Host ""
}
