"""P0③ 修复:退市股财务回填(VIP 按期接口)。

个券财务接口对退市股返回空(镜像实测 603056/600355/000638 全空),但 *_vip 按报告期
全市场接口含退市股(601558 华锐风电实测在内)——按期拉取+分页,把退市股的行拆写进
per-symbol 缓存布局(data/cache/<table>/<code>.parquet),_load/latest_rows 零改动透明受益。
"""
import pandas as pd

from scripts.backfill_delisted import fetch_all_pages, write_missing_symbol_tables


def test_fetch_all_pages_stitches_until_short_page():
    # 模拟 13 行、页大小 5:3 页(5+5+3),末页短即停
    data = pd.DataFrame({"ts_code": [f"c{i}" for i in range(13)], "v": range(13)})
    calls: list[int] = []

    def page(limit: int, offset: int) -> pd.DataFrame:
        calls.append(offset)
        return data.iloc[offset: offset + limit]

    out = fetch_all_pages(page, limit=5)
    assert len(out) == 13 and calls == [0, 5, 10]


def test_fetch_all_pages_empty_first_page():
    out = fetch_all_pages(lambda limit, offset: pd.DataFrame(), limit=5)
    assert out.empty


def test_write_missing_symbol_tables_skips_existing(tmp_path):
    # 已有 per-symbol 缓存的票(在市股)绝不覆盖;缺文件的退市股才写
    (tmp_path / "income").mkdir(parents=True)
    existing = tmp_path / "income" / "600000.SH.parquet"
    pd.DataFrame({"ts_code": ["600000.SH"], "end_date": ["20231231"],
                  "keep": ["old"]}).to_parquet(existing, index=False)
    rows = pd.DataFrame({
        "ts_code": ["600000.SH", "601558.SH", "601558.SH"],
        "end_date": ["20231231", "20180630", "20181231"],
        "keep": ["new", "x", "y"]})
    written = write_missing_symbol_tables(rows, tmp_path, "income")
    assert written == ["601558.SH"]
    assert pd.read_parquet(existing)["keep"].tolist() == ["old"]   # 在市股缓存未被动
    got = pd.read_parquet(tmp_path / "income" / "601558.SH.parquet")
    assert len(got) == 2 and set(got["ts_code"]) == {"601558.SH"}
