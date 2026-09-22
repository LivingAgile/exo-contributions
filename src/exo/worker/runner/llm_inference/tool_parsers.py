import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, cast

from pydantic import JsonValue

from exo.api.types import ToolCallItem


@dataclass
class ToolParser:
    start_parsing: str
    end_parsing: str
    _inner_parser: Callable[[str], list[ToolCallItem] | None]

    def parse(
        self, text: str, tools: list[dict[str, Any]] | None
    ) -> list[ToolCallItem] | None:
        parsed = self._inner_parser(text)
        if parsed is None:
            return None
        if tools is not None:
            parsed = _coerce_tool_calls_to_schema(parsed, tools)
        return parsed


def _json_type_matches(value: Any, expected_type: str) -> bool:  # pyright: ignore[reportAny]
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(
            value, float
        )
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return False


def _coerce_tool_arg_with_schema(value: Any, schema: dict[str, Any]) -> Any:  # pyright: ignore[reportAny]
    schema_type = schema.get("type")

    if isinstance(schema_type, list):
        for candidate in schema_type:  # pyright: ignore[reportUnknownVariableType]
            if not isinstance(candidate, str):
                continue
            if candidate == "null" and value is None:
                return None
            candidate_schema = {**schema, "type": candidate}
            coerced = _coerce_tool_arg_with_schema(value, candidate_schema)  # pyright: ignore[reportAny]
            if _json_type_matches(coerced, candidate):
                return coerced  # pyright: ignore[reportAny]
        return value  # pyright: ignore[reportAny]

    if not isinstance(schema_type, str):
        return value  # pyright: ignore[reportAny]

    if schema_type == "object":
        parsed = value  # pyright: ignore[reportAny]
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)  # pyright: ignore[reportAny]
            except Exception:
                return value  # pyright: ignore[reportAny]
        if not isinstance(parsed, dict):
            return value  # pyright: ignore[reportAny]
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return parsed  # pyright: ignore[reportUnknownVariableType]
        return {
            key: (
                _coerce_tool_arg_with_schema(prop_value, prop_schema)  # pyright: ignore[reportUnknownArgumentType]
                if isinstance(prop_schema, dict)
                else prop_value
            )
            for key, prop_value in parsed.items()  # pyright: ignore[reportUnknownVariableType]
            for prop_schema in [properties.get(key)]  # type: ignore
        }

    if schema_type == "array":
        parsed = value  # pyright: ignore[reportAny]
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)  # pyright: ignore[reportAny]
            except Exception:
                return value  # pyright: ignore[reportAny]
        if not isinstance(parsed, list):
            return value  # pyright: ignore[reportAny]
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            return parsed  # pyright: ignore[reportUnknownVariableType]
        return [_coerce_tool_arg_with_schema(item, item_schema) for item in parsed]  # type: ignore

    if schema_type == "integer":
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                return value
        return value

    if schema_type == "number":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            try:
                num = float(value.strip())
                if math.isfinite(num):
                    return num
            except ValueError:
                return value
        return value

    if schema_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
        return value

    return value  # pyright: ignore[reportAny]


def _coerce_tool_calls_to_schema(
    tool_calls: list[ToolCallItem], tools: list[dict[str, Any]]
) -> list[ToolCallItem]:
    schema_by_name: dict[str, dict[str, Any]] = {}
    for tool in tools:
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")  # type: ignore
        parameters = function.get("parameters")  # type: ignore
        if isinstance(name, str) and isinstance(parameters, dict):
            schema_by_name[name] = parameters

    if not schema_by_name:
        return tool_calls

    coerced_calls: list[ToolCallItem] = []
    for tool_call in tool_calls:
        schema = schema_by_name.get(tool_call.name)
        if schema is None:
            coerced_calls.append(tool_call)
            continue

        try:
            parsed_args = json.loads(tool_call.arguments)  # pyright: ignore[reportAny]
        except Exception:
            coerced_calls.append(tool_call)
            continue

        if not isinstance(parsed_args, dict):
            coerced_calls.append(tool_call)
            continue

        coerced_args = _coerce_tool_arg_with_schema(parsed_args, schema)  # pyright: ignore[reportAny]
        if not isinstance(coerced_args, dict):
            coerced_calls.append(tool_call)
            continue

        coerced_calls.append(
            tool_call.model_copy(update={"arguments": json.dumps(coerced_args)})
        )
    return coerced_calls


def make_mlx_parser(
    tool_call_start: str,
    tool_call_end: str,
    tool_parser: Callable[[str], dict[str, Any] | list[dict[str, Any]]],
) -> ToolParser:
    def parse_tool_calls(text: str) -> list[ToolCallItem] | None:
        try:
            text = text.removeprefix(tool_call_start)
            text = text.removesuffix(tool_call_end)
            parsed = tool_parser(text)
            if isinstance(parsed, list):
                return [ToolCallItem.model_validate(_flatten(p)) for p in parsed]
            else:
                return [ToolCallItem.model_validate(_flatten(parsed))]

        except Exception:
            return None

    return ToolParser(
        start_parsing=tool_call_start,
        end_parsing=tool_call_end,
        _inner_parser=parse_tool_calls,
    )


# TODO / example code:
def _parse_json_calls(text: str) -> list[ToolCallItem] | None:
    try:
        text = text.removeprefix("<tool_call>")
        text = text.removesuffix("</tool_call>")
        top_level = {
            k: json.dumps(v) if isinstance(v, (dict, list)) else v
            for k, v in json.loads(text).items()  # pyright: ignore[reportAny]
        }
        return [ToolCallItem.model_validate(top_level)]
    except Exception:
        return None


def _flatten(p: dict[str, Any]) -> dict[str, str]:
    return {
        k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)  # pyright: ignore[reportAny]
        for k, v in p.items()  # pyright: ignore[reportAny]
    }


def make_json_parser() -> ToolParser:
    return ToolParser(
        start_parsing="<tool_call>",
        end_parsing="</tool_call>",
        _inner_parser=_parse_json_calls,
    )


def parse_atem_tool_calls(
    text: str,
    recipient: str,
    tools: list[dict[str, Any]] | None,
) -> list[ToolCallItem] | None:
    if tools is None:
        return None

    schemas: dict[str, dict[str, JsonValue]] = {}
    for tool in tools:
        function = cast(dict[str, JsonValue], tool).get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        parameters = function.get("parameters")
        if isinstance(name, str) and isinstance(parameters, dict):
            schemas[name] = parameters

    block_match = re.fullmatch(
        r"\s*<atem:function_calls>(.*?)</atem:function_calls>\s*",
        text,
        re.DOTALL,
    )
    if block_match is None:
        return None

    calls_text = block_match.group(1)
    invoke_pattern = re.compile(
        r'\s*<atem:invoke\s+name="([^"]+)">(.*?)</atem:invoke>',
        re.DOTALL,
    )
    parameter_pattern = re.compile(
        r'\s*<atem:parameter\s+name="([^"]+)">(.*?)</atem:parameter>',
        re.DOTALL,
    )
    calls: list[ToolCallItem] = []
    position = 0

    for invoke in invoke_pattern.finditer(calls_text):
        if invoke.start() != position and calls_text[position : invoke.start()].strip():
            return None
        position = invoke.end()
        name = invoke.group(1)
        schema = schemas.get(name)
        if not name or schema is None or not _atem_recipient_matches(recipient, name):
            return None

        properties = schema.get("properties")
        if not isinstance(properties, dict):
            properties = {}
        arguments: dict[str, object] = {}
        parameters_text = invoke.group(2)
        parameter_position = 0
        for parameter in parameter_pattern.finditer(parameters_text):
            if (
                parameter.start() != parameter_position
                and parameters_text[parameter_position : parameter.start()].strip()
            ):
                return None
            parameter_position = parameter.end()
            parameter_name = parameter.group(1)
            if parameter_name in arguments:
                return None
            parameter_schema = properties.get(parameter_name)
            if not isinstance(parameter_schema, dict):
                return None
            converted = _parse_atem_parameter(parameter.group(2), parameter_schema)
            if converted is _INVALID_ATEM_VALUE:
                return None
            arguments[parameter_name] = converted

        if parameters_text[parameter_position:].strip():
            return None
        required = schema.get("required", [])
        if not isinstance(required, list) or any(
            not isinstance(item, str) or item not in arguments for item in required
        ):
            return None
        if schema.get("additionalProperties") is False and any(
            key not in properties for key in arguments
        ):
            return None
        calls.append(
            ToolCallItem(
                name=name,
                arguments=json.dumps(arguments, separators=(",", ":")),
            )
        )

    if calls_text[position:].strip() or not calls:
        return None
    return calls


_INVALID_ATEM_VALUE = object()


def _atem_recipient_matches(recipient: str, function_name: str) -> bool:
    if recipient in (function_name, "functions"):
        return True
    namespace, separator, _ = function_name.rpartition(".")
    return bool(separator) and recipient in (namespace, f"{namespace}.*")


def _parse_atem_parameter(text: str, schema: dict[str, JsonValue]) -> object:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        for candidate in schema_type:
            if not isinstance(candidate, str):
                continue
            converted = _parse_atem_parameter(text, {**schema, "type": candidate})
            if converted is not _INVALID_ATEM_VALUE:
                return converted
        return _INVALID_ATEM_VALUE

    if schema_type == "string":
        converted: object = text
    elif schema_type == "integer":
        if re.fullmatch(r"[+-]?\d+", text.strip()) is None:
            return _INVALID_ATEM_VALUE
        converted = int(text.strip())
    elif schema_type == "number":
        try:
            converted = float(text.strip())
        except ValueError:
            return _INVALID_ATEM_VALUE
        if not math.isfinite(converted):
            return _INVALID_ATEM_VALUE
    elif schema_type == "boolean":
        if text.strip().lower() not in ("true", "false"):
            return _INVALID_ATEM_VALUE
        converted = text.strip().lower() == "true"
    elif schema_type == "null":
        if text.strip() != "null":
            return _INVALID_ATEM_VALUE
        converted = None
    elif schema_type in ("array", "object"):
        try:
            parsed = cast(JsonValue, json.loads(text))
        except Exception:
            return _INVALID_ATEM_VALUE
        converted = _validate_atem_json_value(parsed, schema)
        if converted is _INVALID_ATEM_VALUE:
            return _INVALID_ATEM_VALUE
    else:
        return _INVALID_ATEM_VALUE

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and converted not in enum_values:
        return _INVALID_ATEM_VALUE
    return converted


def _validate_atem_json_value(value: JsonValue, schema: dict[str, JsonValue]) -> object:
    schema_type = schema.get("type")
    if not isinstance(schema_type, str) or not _json_type_matches(value, schema_type):
        return _INVALID_ATEM_VALUE
    if schema_type == "array":
        if not isinstance(value, list):
            return _INVALID_ATEM_VALUE
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            return value
        converted_items = [
            _validate_atem_json_value(item, item_schema) for item in value
        ]
        if any(item is _INVALID_ATEM_VALUE for item in converted_items):
            return _INVALID_ATEM_VALUE
        return converted_items
    if schema_type == "object":
        if not isinstance(value, dict):
            return _INVALID_ATEM_VALUE
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return _INVALID_ATEM_VALUE
        required = schema.get("required", [])
        if not isinstance(required, list) or any(
            not isinstance(key, str) or key not in value for key in required
        ):
            return _INVALID_ATEM_VALUE
        if schema.get("additionalProperties") is False and any(
            key not in properties for key in value
        ):
            return _INVALID_ATEM_VALUE
        converted_object: dict[str, object] = {}
        for key, item in value.items():
            item_schema = properties.get(key)
            if not isinstance(item_schema, dict):
                converted_object[key] = item
                continue
            converted_item = _validate_atem_json_value(item, item_schema)
            if converted_item is _INVALID_ATEM_VALUE:
                return _INVALID_ATEM_VALUE
            converted_object[key] = converted_item
        return converted_object
    return value


def infer_tool_parser(chat_template: str) -> ToolParser | None:
    """Attempt to auto-infer a tool parser from the chat template."""
    if "<tool_call>" in chat_template and "tool_call.name" in chat_template:
        return make_json_parser()
    return None
