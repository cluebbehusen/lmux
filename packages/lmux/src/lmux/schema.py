"""JSON Schema utilities for provider mappers."""

from typing import Any, cast


def add_additional_properties_false(schema: dict[str, Any]) -> None:
    """Recursively set ``additionalProperties: false`` on all object-typed nodes.

    Some LLM APIs require this field on every ``object`` node in a JSON Schema.
    Many schema generators omit it, so providers can call this to patch in-place
    before sending.
    """
    if schema.get("type") == "object" and "additionalProperties" not in schema:
        schema["additionalProperties"] = False
    for value in schema.values():
        if isinstance(value, dict):
            add_additional_properties_false(cast("dict[str, Any]", value))
        elif isinstance(value, list):
            for item in cast("list[Any]", value):
                if isinstance(item, dict):
                    add_additional_properties_false(cast("dict[str, Any]", item))
