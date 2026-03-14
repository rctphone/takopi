from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
import os
from pathlib import Path
import tempfile
from typing import Any

import tomli_w

HOME_CONFIG_PATH = Path.home() / ".takopi" / "takopi.toml"


class ConfigError(RuntimeError):
    pass


def ensure_table(
    config: dict[str, Any],
    key: str,
    *,
    config_path: Path,
    label: str | None = None,
) -> dict[str, Any]:
    value = config.get(key)
    if value is None:
        table: dict[str, Any] = {}
        config[key] = table
        return table
    if not isinstance(value, dict):
        name = label or key
        raise ConfigError(f"Invalid `{name}` in {config_path}; expected a table.")
    return value


def read_config(cfg_path: Path) -> dict:
    if cfg_path.exists() and not cfg_path.is_file():
        raise ConfigError(f"Config path {cfg_path} exists but is not a file.") from None
    try:
        raw = cfg_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"Missing config file {cfg_path}.") from None
    except OSError as e:
        raise ConfigError(f"Failed to read config file {cfg_path}: {e}") from e
    try:
        return tomllib.loads(raw)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Malformed TOML in {cfg_path}: {e}") from None


def load_or_init_config(path: str | Path | None = None) -> tuple[dict, Path]:
    cfg_path = Path(path).expanduser() if path else HOME_CONFIG_PATH
    if cfg_path.exists() and not cfg_path.is_file():
        raise ConfigError(f"Config path {cfg_path} exists but is not a file.") from None
    if not cfg_path.exists():
        return {}, cfg_path
    return read_config(cfg_path), cfg_path


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    alias: str
    path: Path
    worktrees_dir: Path
    default_engine: str | None = None
    worktree_base: str | None = None
    chat_id: int | None = None
    topic_id: int | None = None
    default_trigger_mode: str | None = None
    allowed_tools: list[str] | None = None

    @property
    def has_dedicated_chat(self) -> bool:
        return self.chat_id is not None

    @property
    def worktrees_root(self) -> Path:
        if self.worktrees_dir.is_absolute():
            return self.worktrees_dir
        return self.path / self.worktrees_dir


@dataclass(frozen=True, slots=True)
class ProjectsConfig:
    projects: dict[str, ProjectConfig]
    default_project: str | None = None
    chat_map: dict[int, str] = field(default_factory=dict)
    topic_map: dict[tuple[int, int], str] = field(default_factory=dict)

    def resolve(self, alias: str | None) -> ProjectConfig | None:
        if alias is None:
            if self.default_project is None:
                return None
            return self.projects.get(self.default_project)
        return self.projects.get(alias.lower())

    def project_for_chat(self, chat_id: int | None) -> str | None:
        if chat_id is None:
            return None
        return self.chat_map.get(chat_id)

    def project_for_topic(self, chat_id: int, thread_id: int | None) -> str | None:
        if thread_id is None:
            return None
        return self.topic_map.get((chat_id, thread_id))

    def trigger_mode_for_chat(
        self, chat_id: int | None, thread_id: int | None = None,
    ) -> str | None:
        if chat_id is not None:
            for key in (
                self.project_for_topic(chat_id, thread_id) if thread_id else None,
                self.project_for_chat(chat_id),
            ):
                if key is not None:
                    p = self.projects.get(key)
                    if p is not None and p.default_trigger_mode is not None:
                        return p.default_trigger_mode
        return None

    def project_chat_ids(self) -> tuple[int, ...]:
        ids = set(self.chat_map.keys())
        ids.update(cid for cid, _ in self.topic_map.keys())
        return tuple(sorted(ids))


def dump_toml(config: dict[str, Any]) -> str:
    try:
        dumped = tomli_w.dumps(config)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"Unsupported config value: {e}") from None
    return dumped


def write_config(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dump_toml(config)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(payload)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
    except OSError as e:
        raise ConfigError(f"Failed to write config file {path}: {e}") from e
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
