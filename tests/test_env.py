"""Tests for .env.local loading — 必须是权威配置(override 掉进程里残留的旧 env)。"""
import os

from ashare_gauntlet.data.env import load_env_local


def test_load_env_local_overrides_inherited_env(tmp_path, monkeypatch):
    # .env.local 是项目数据源的权威配置:换源后即便进程里残留旧 TUSHARE_*,
    # .env.local 的新值也要生效——否则"改了 .env.local 等于没改"(实际踩过的坑)。
    (tmp_path / ".env.local").write_text(
        'TUSHARE_TOKEN=newtoken\nTUSHARE_HTTP_URL="https://new.example"\n', encoding="utf-8"
    )
    monkeypatch.setenv("TUSHARE_TOKEN", "oldtoken")
    monkeypatch.setenv("TUSHARE_HTTP_URL", "http://old")

    load_env_local(tmp_path / ".env.local")

    assert os.environ["TUSHARE_TOKEN"] == "newtoken"
    assert os.environ["TUSHARE_HTTP_URL"] == "https://new.example"  # 引号被剥掉


def test_load_env_local_skips_comments_and_blank_lines(tmp_path, monkeypatch):
    (tmp_path / ".env.local").write_text("# 注释行\n\nFOO=bar\n", encoding="utf-8")
    monkeypatch.delenv("FOO", raising=False)

    load_env_local(tmp_path / ".env.local")

    assert os.environ["FOO"] == "bar"


def test_load_env_local_noop_when_file_missing(tmp_path):
    # 文件不存在时安静返回,不抛(调用方无需先判存在)。
    load_env_local(tmp_path / "nonexistent.env")
