import json
from collections.abc import Generator

from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.worker.runner.llm_inference.model_output_parsers import parse_deepseek_v41


def _simulate_tokens(tokens: list[str]) -> Generator[GenerationResponse | None]:
    for index, token in enumerate(tokens):
        yield GenerationResponse(
            text=token,
            token=index,
            finish_reason="stop" if index == len(tokens) - 1 else None,
            usage=None,
        )


def test_v41_leading_space_dsml_tags_are_extracted_across_tokens() -> None:
    dsml = "｜DSML｜"
    model_tokens = [
        "answer",
        "\n\n<",
        dsml,
        " calls>",
        "\n<",
        dsml,
        ' invoke name="search::lookup">\n',
        "<",
        dsml,
        ' parameter name="limit" string="false">3</',
        dsml,
        " parameter>\n</",
        dsml,
        " invoke>\n</",
        dsml,
        " calls>",
    ]

    results = list(parse_deepseek_v41(_simulate_tokens(model_tokens)))
    tool_results = [
        result for result in results if isinstance(result, ToolCallResponse)
    ]
    text = "".join(
        result.text for result in results if isinstance(result, GenerationResponse)
    )

    assert text == "answer\n\n"
    assert len(tool_results) == 1
    assert tool_results[0].tool_calls[0].name == "search::lookup"
    assert json.loads(tool_results[0].tool_calls[0].arguments) == {"limit": 3}
