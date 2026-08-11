"""项目本地配置加载:把 ``.env.local`` 当**权威**数据源配置读进 ``os.environ``。

为什么是 override 而不是 setdefault:换数据源时只改 ``.env.local`` 是用户的心智
模型,但若进程里已残留旧的 ``TUSHARE_TOKEN`` / ``TUSHARE_HTTP_URL``(例如调度任务
继承了 app 启动时的旧环境),``setdefault`` 不会覆盖 → "改了 .env.local 等于没改"。
所以这里 override:``.env.local`` 存在即以它为准。它只覆盖文件里**显式列出**的键,
不动其它环境变量(如全局 PYTHONUTF8)。
"""
from __future__ import annotations

import os
from pathlib import Path


def load_env_local(path: str | Path = ".env.local", *,
                   allowed_keys: set[str] | None = None) -> None:
    """把 ``.env.local`` 里的 ``KEY=VALUE`` 读入并**覆盖** ``os.environ``。

    文件不存在时安静返回(调用方无需先判存在)。跳过空行与 ``#`` 注释,
    剥掉值两侧的成对引号。
    """
    p = Path(path)
    if not p.exists():
        return
    parsed: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, value = s.split("=", 1)
        key = key.strip()
        if allowed_keys is not None and key not in allowed_keys:
            raise ValueError(f"unsupported key in {p}: {key}")
        parsed[key] = value.strip().strip('"').strip("'")
    os.environ.update(parsed)
