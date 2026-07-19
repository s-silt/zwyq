"""namechange 全量拉取 → data/cache/namechange/all.parquet(X-05 数据工程)。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.backfill_namechange

分页拉取 tushare `namechange`(证券曾用名/变更区间)。脏行(start_date 非
YYYYMMDD)在此层**对着数字**剔除并打印明细——loader(namechange.load_namechange)
对缓存零容忍,清理责任全在本层,不藏不补。幂等:重跑整体覆盖。
"""
from __future__ import annotations

import os

import pandas as pd

from ashare_gauntlet.config import CACHE_DIR as CACHE, tushare_pro
from ashare_gauntlet.data.fetch import call_with_retry

PAGE = 5000   # 纯分页机械参数(单页行数),非研究常数


def main() -> None:
    pro = tushare_pro()
    frames: list[pd.DataFrame] = []
    offset = 0
    while True:
        df = call_with_retry(lambda o=offset: pro.namechange(
            limit=PAGE, offset=o,
            fields="ts_code,name,start_date,end_date,ann_date,change_reason"))
        if df is None or df.empty:
            break
        frames.append(df)
        if len(df) < PAGE:
            break
        offset += PAGE
    if not frames:
        raise SystemExit("namechange 拉取为空——检查 token/接口权限,不落空文件")
    allrows = pd.concat(frames, ignore_index=True).drop_duplicates()
    bad = ~allrows["start_date"].astype(str).str.fullmatch(r"\d{8}", na=True) | \
        allrows["start_date"].isna()
    if bool(bad.any()):
        print(f"⚠ 剔除 {int(bad.sum())} 行无效 start_date(无法定位时间轴):")
        print(allrows.loc[bad, ["ts_code", "name", "start_date", "ann_date"]]
              .to_string(index=False))
        allrows = allrows[~bad]
    if allrows.empty:
        raise SystemExit("清理后 namechange 为空——拒绝落盘")
    os.makedirs(f"{CACHE}/namechange", exist_ok=True)
    allrows.to_parquet(f"{CACHE}/namechange/all.parquet", index=False)
    st_rows = allrows["name"].astype(str).str.contains("ST", na=False)
    print(f"namechange 落盘:{len(allrows)} 行 / {allrows['ts_code'].nunique()} 只;"
          f"含 ST 名称行 {int(st_rows.sum())};区间 "
          f"{allrows['start_date'].min()} → {allrows['start_date'].max()}"
          f" → {CACHE}/namechange/all.parquet")


if __name__ == "__main__":
    main()
