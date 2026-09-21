from pathlib import Path


def main() -> None:
    path = Path(__file__).with_name("run_judge_development_direct_transport.py")
    text = path.read_text(encoding="utf-8")

    assert 'schema_name == "CompositeEvidenceVerification"' in text
    assert 'output_format: Any = "json" if composite_json_mode else schema.model_json_schema()' in text
    assert '"num_predict": self.num_predict' in text
    assert 'starting schema={schema_name}' in text
    assert 'return result' in text
    assert 'schema.model_validate_json(content)' not in text

    print("v0.23 direct-transport contract tests passed.")


if __name__ == "__main__":
    main()
