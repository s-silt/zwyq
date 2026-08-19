"""账本排他锁:holdings.json + trade_journal.json 的**全部写入方**共用。

单一锁文件(<holdings>.lock)覆盖整个账户账本(两个文件视为一本账):
trade_record / holdings_confirm / trade_journal --add 写前必须持锁,
否则并发写会互相覆盖(codex 复审 P0:锁只被一个写入方遵守=没有锁)。
读方不加锁(JSON 原子替换保证读到的是完整旧版或完整新版)。
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path


class AccountLockError(SystemExit):
    """账本被另一次写入占用。"""


@contextmanager
def account_lock(holdings_path: str | os.PathLike[str]):
    """排他锁:O_CREAT|O_EXCL 抢占;失败即 fail-loud,不等待不重试。"""
    lock_path = Path(holdings_path).with_suffix(".lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise AccountLockError(
            f"检测到账本锁 {lock_path}——另一次写入进行中;"
            "确认无其他 trade_record/holdings_confirm/trade_journal 进程后手动删除再重试")
    try:
        yield
    finally:
        os.close(fd)
        try:
            os.unlink(lock_path)
        except OSError:
            pass
