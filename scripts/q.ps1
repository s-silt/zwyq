# q.ps1 — ashare-gauntlet 统一入口:把常用命令收敛成短动词。
# 只读命令直呈;账本写命令(confirm/ledger)仍是"运行=人工签字",不代跑代判、不下单。
# 用法:  .\scripts\q.ps1 <verb> [args...]
#   close   <YYYYMMDD>      收盘一条龙:EOD→人工确认→估值→一屏 -> scripts.eod_close
#   today   [--json]        每日一屏简报(只读)            -> scripts.daily_brief
#   eod     [--skip-probe]  每日 EOD 编排(刷新→排名→决策) -> scripts.eod_ops
#   watch   [--dedupe]      盘中哨兵                         -> scripts.intraday_watch
#   temp                    市场温度一行                     -> scripts.market_temp
#   confirm <YYYYMMDD>      推进账户 as_of(人工签字)        -> scripts.holdings_confirm
#   ledger  <args...>       人工成交落账(人工签字)          -> scripts.trade_record
#   help                    本帮助
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot   # scripts/ 的上级 = 仓库根
$py = Join-Path $root ".venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"

$verb = if ($args.Count -ge 1) { [string]$args[0] } else { "help" }
$rest = if ($args.Count -ge 2) { $args[1..($args.Count - 1)] } else { @() }

$map = @{
  today   = "scripts.daily_brief"
  brief   = "scripts.daily_brief"
  eod     = "scripts.eod_ops"
  watch   = "scripts.intraday_watch"
  temp    = "scripts.market_temp"
  confirm = "scripts.holdings_confirm"
  ledger  = "scripts.trade_record"
  audit   = "scripts.ledger_reconcile"
  gate    = "scripts.gate_check"
  close   = "scripts.eod_close"
}

function Show-Help {
  Write-Host "q — ashare-gauntlet 统一入口"
  Write-Host "  q close   <YYYYMMDD>      收盘一条龙:EOD→人工确认→估值→一屏"
  Write-Host "  q today   [--json]        每日一屏简报(只读)"
  Write-Host "  q eod     [--skip-probe]  每日 EOD 编排"
  Write-Host "  q watch   [--dedupe]      盘中哨兵"
  Write-Host "  q temp                    市场温度一行"
  Write-Host "  q confirm <YYYYMMDD>      推进账户 as_of(人工签字)"
  Write-Host "  q ledger  <args...>       人工成交落账(人工签字;支持 --advance-as-of/--trim)"
  Write-Host "  q audit   [--json]        账本对账(只读):holdings ↔ trade_journal"
  Write-Host "  q gate    [--freeze]      门禁证据体检(季度):五门+组合t vs 冻结基线"
  Write-Host "  q help                    本帮助"
}

if ($verb -in @("help", "-h", "--help")) { Show-Help; exit 0 }

if (-not $map.ContainsKey($verb)) {
  Write-Host "未知命令: $verb`n"
  Show-Help
  exit 64
}

if (-not (Test-Path $py)) {
  Write-Host "找不到解释器: $py(确认 .venv 已建)"
  exit 69
}

Push-Location $root
try {
  & $py -m $map[$verb] @rest
  $code = $LASTEXITCODE
} finally {
  Pop-Location
}
exit $code
