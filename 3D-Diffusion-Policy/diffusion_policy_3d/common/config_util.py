from typing import Any, List, Optional


def cfg_path_get(root: Any, path: str, default: Any = None) -> Any:
    current = root
    for key in path.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key, None)
        elif hasattr(current, "get"):
            current = current.get(key, None)
        else:
            current = getattr(current, key, None)
    return default if current is None else current


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
    return bool(value)


def get_task_execution_mode(cfg: Any, default: str = "sim") -> str:
    value = cfg_path_get(cfg, "task.execution.mode", None)
    if value is None:
        return default
    return str(value)


def get_task_execution_flag(cfg: Any, key: str, default: bool) -> bool:
    value = cfg_path_get(cfg, f"task.execution.{key}", None)
    return _coerce_bool(value, default)


def is_eval_enabled(cfg: Any) -> bool:
    return get_task_execution_flag(cfg, "enable_eval", True)


def is_amq_enabled(cfg: Any) -> bool:
    return get_task_execution_flag(cfg, "enable_amq", True)


def is_cm_policy_enabled(cfg: Any) -> bool:
    return get_task_execution_flag(cfg, "enable_cm_policy", True)


def stop_env_on_keyboard_interrupt(cfg: Any) -> bool:
    return get_task_execution_flag(cfg, "stop_env_on_keyboard_interrupt", False)


def get_final_eval_policies(
    cfg: Any,
    default: Optional[List[str]] = None,
) -> List[str]:
    if default is None:
        default = ["ddim"]

    value = cfg_path_get(cfg, "runtime.final_eval_policies", default)
    policies = [str(policy).lower() for policy in list(value)]
    if not is_cm_policy_enabled(cfg):
        policies = [policy for policy in policies if policy != "cm"]
    return policies
