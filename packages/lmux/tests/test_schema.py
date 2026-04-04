"""Tests for JSON Schema utilities."""

from typing import Any, cast

from lmux.schema import add_additional_properties_false


class TestAddAdditionalPropertiesFalse:
    def test_adds_to_top_level_object(self) -> None:
        schema: dict[str, object] = {"type": "object", "properties": {"name": {"type": "string"}}}
        add_additional_properties_false(schema)
        assert schema["additionalProperties"] is False

    def test_does_not_overwrite_existing(self) -> None:
        schema: dict[str, object] = {"type": "object", "additionalProperties": True}
        add_additional_properties_false(schema)
        assert schema["additionalProperties"] is True

    def test_skips_non_object_types(self) -> None:
        schema: dict[str, object] = {"type": "string"}
        add_additional_properties_false(schema)
        assert "additionalProperties" not in schema

    def test_recurses_into_nested_objects(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {"street": {"type": "string"}},
                }
            },
        }
        add_additional_properties_false(schema)
        assert schema["additionalProperties"] is False
        address = cast("dict[str, Any]", cast("dict[str, Any]", schema["properties"])["address"])
        assert address["additionalProperties"] is False

    def test_recurses_into_array_items(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "tags": {
                    "anyOf": [
                        {"type": "object", "properties": {"label": {"type": "string"}}},
                    ]
                }
            },
        }
        add_additional_properties_false(schema)
        tags = cast("dict[str, Any]", cast("dict[str, Any]", schema["properties"])["tags"])
        inner = cast("dict[str, Any]", cast("list[Any]", tags["anyOf"])[0])
        assert inner["additionalProperties"] is False

    def test_skips_non_dict_array_items(self) -> None:
        schema: dict[str, object] = {"type": "object", "required": ["name", "age"]}
        add_additional_properties_false(schema)
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["name", "age"]
