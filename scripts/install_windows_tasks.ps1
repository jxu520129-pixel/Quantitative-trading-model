param(
    [string]$PythonPath = "$PSScriptRoot\..\.venv\Scripts\python.exe",
    [string]$ProjectPath = (Resolve-Path "$PSScriptRoot\..").Path,
    [ValidateSet("Scheduler", "Discrete")]
    [string]$Mode = "Scheduler"
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "未找到 Python 可执行文件：$PythonPath"
}

if ($Mode -eq "Scheduler") {
    $action = New-ScheduledTaskAction -Execute $PythonPath -Argument "-m ashare_quant.cli scheduler" -WorkingDirectory $ProjectPath
    $trigger = New-ScheduledTaskTrigger -AtStartup
    Register-ScheduledTask -TaskName "A股量化交易-调度器" -Action $action -Trigger $trigger -Description "长期运行的 A 股量化交易调度器" -Force
    Write-Output "已创建计划任务：A股量化交易-调度器。"
} else {
    $jobs = @(
        @{ Name = "A股量化交易-晨间成交"; Time = "09:35"; Command = "execute-orders"; Description = "A 股量化交易晨间模拟成交" },
        @{ Name = "A股量化交易-盘后信号"; Time = "15:20"; Command = "after-close"; Description = "A 股量化交易盘后数据、信号与委托排队" },
        @{ Name = "A股量化交易-收盘风控"; Time = "15:35"; Command = "end-of-day"; Description = "A 股量化交易收盘估值与风控" }
    )
    foreach ($job in $jobs) {
        $action = New-ScheduledTaskAction -Execute $PythonPath -Argument "-m ashare_quant.cli $($job.Command)" -WorkingDirectory $ProjectPath
        $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $job.Time
        Register-ScheduledTask -TaskName $job.Name -Action $action -Trigger $trigger -Description $job.Description -Force
    }
    Write-Output "已创建三个分时计划任务。"
}
