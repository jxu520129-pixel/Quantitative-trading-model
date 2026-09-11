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
    # 与 config/default.yaml 的 scheduler 时点保持一致。
    # 盘中 intraday-trade 是**买点的唯一入口**（实时价当天成交），少了它就不会有任何买入。
    $jobs = @(
        @{ Name = "A股量化交易-开盘前结算"; Time = "09:26"; Command = "execute-orders"; Description = "T+1 解锁可卖数量与风控日内状态重置" },
        @{ Name = "A股量化交易-盘中即时交易-0937"; Time = "09:37"; Command = "intraday-trade"; Description = "盘中实时价即时买卖" },
        @{ Name = "A股量化交易-盘中即时交易-1000"; Time = "10:00"; Command = "intraday-trade"; Description = "盘中实时价即时买卖" },
        @{ Name = "A股量化交易-盘中即时交易-1030"; Time = "10:30"; Command = "intraday-trade"; Description = "盘中实时价即时买卖" },
        @{ Name = "A股量化交易-盘中即时交易-1430"; Time = "14:30"; Command = "intraday-trade"; Description = "盘中实时价即时买卖" },
        @{ Name = "A股量化交易-盘中即时交易-1450"; Time = "14:50"; Command = "intraday-trade"; Description = "盘中实时价即时买卖" },
        @{ Name = "A股量化交易-盘后离场"; Time = "15:30"; Command = "after-close"; Description = "更新日线、买点扫描与按收盘价离场" },
        @{ Name = "A股量化交易-收盘风控"; Time = "15:50"; Command = "end-of-day"; Description = "收盘估值与风控" }
    )
    foreach ($job in $jobs) {
        $action = New-ScheduledTaskAction -Execute $PythonPath -Argument "-m ashare_quant.cli $($job.Command)" -WorkingDirectory $ProjectPath
        $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $job.Time
        Register-ScheduledTask -TaskName $job.Name -Action $action -Trigger $trigger -Description $job.Description -Force
    }
    Write-Output "已创建 $($jobs.Count) 个分时计划任务。"
}
