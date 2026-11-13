from click.testing import CliRunner
import httpx
import json
from llm.cli import cli
import pytest


@pytest.fixture
def invoke_files(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))

    def invoke(pages):
        requests = []

        def handle(request):
            assert request.method == "GET"
            assert request.url.host == "generativelanguage.googleapis.com"
            assert request.url.path == "/v1beta/files"
            assert request.url.params["key"] == "test-key"
            token = request.url.params.get("pageToken")
            requests.append(token)
            page = pages[token]
            if isinstance(page, int):
                return httpx.Response(page, json={"error": {"message": "Unavailable"}})
            return httpx.Response(200, json=page)

        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            monkeypatch.setattr("llm_gemini.httpx.get", client.get)
            result = CliRunner().invoke(cli, ["gemini", "files", "--key", "test-key"])
        return result, requests

    return invoke


def test_files_pagination(invoke_files):
    first = {"name": "files/first", "displayName": "First file"}
    second = {"name": "files/second"}
    last = {"name": "files/last"}
    result, requests = invoke_files(
        {
            None: {"files": [first], "nextPageToken": "second/+?="},
            "second/+?=": {"files": [second], "nextPageToken": "third/+?="},
            "third/+?=": {"files": [last]},
        }
    )
    assert result.exit_code == 0, result.exception
    assert json.loads(result.stdout) == [first, second, last]
    assert result.stderr == ""
    assert requests == [None, "second/+?=", "third/+?="]


@pytest.mark.parametrize("empty_page", [{}, {"files": []}])
def test_files_empty_intermediate_page(invoke_files, empty_page):
    first = {"name": "files/first"}
    last = {"name": "files/last"}
    result, requests = invoke_files(
        {
            None: {"files": [first], "nextPageToken": "second"},
            "second": {**empty_page, "nextPageToken": "third"},
            "third": {"files": [last]},
        }
    )
    assert result.exit_code == 0, result.exception
    assert json.loads(result.stdout) == [first, last]
    assert result.stderr == ""
    assert requests == [None, "second", "third"]


@pytest.mark.parametrize(
    "page,expected_stdout,expected_stderr",
    [
        ({}, "", "No files uploaded to the Gemini API.\n"),
        ({"files": []}, "[]\n", ""),
        (
            {"files": [{"name": "files/only"}]},
            '[\n  {\n    "name": "files/only"\n  }\n]\n',
            "",
        ),
    ],
)
def test_files_single_page_compatibility(
    invoke_files, page, expected_stdout, expected_stderr
):
    result, requests = invoke_files({None: page})
    assert result.exit_code == 0, result.exception
    assert result.stdout == expected_stdout
    assert result.stderr == expected_stderr
    assert requests == [None]


def test_files_empty_first_page(invoke_files):
    last = {"name": "files/last"}
    result, requests = invoke_files(
        {None: {"nextPageToken": "second"}, "second": {"files": [last]}}
    )
    assert result.exit_code == 0, result.exception
    assert json.loads(result.stdout) == [last]
    assert result.stderr == ""
    assert requests == [None, "second"]


@pytest.mark.parametrize(
    "first,last,expected_stdout,expected_stderr",
    [
        ({}, {}, "", "No files uploaded to the Gemini API.\n"),
        ({"files": []}, {}, "[]\n", ""),
        ({}, {"files": []}, "[]\n", ""),
    ],
)
def test_files_all_pages_empty(
    invoke_files, first, last, expected_stdout, expected_stderr
):
    result, requests = invoke_files(
        {None: {**first, "nextPageToken": "second"}, "second": last}
    )
    assert result.exit_code == 0, result.exception
    assert result.stdout == expected_stdout
    assert result.stderr == expected_stderr
    assert requests == [None, "second"]


@pytest.mark.parametrize("status", [401, 503])
def test_files_later_page_error(invoke_files, status):
    result, requests = invoke_files(
        {
            None: {"files": [{"name": "files/first"}], "nextPageToken": "second"},
            "second": status,
        }
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, httpx.HTTPStatusError)
    assert result.exception.response.status_code == status
    assert result.stdout == ""
    assert result.stderr == ""
    assert requests == [None, "second"]
