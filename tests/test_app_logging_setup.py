"""The configured log file receives the log at every level.

Every install's server-config carries ``log_file`` (``Server.init_server_config``
writes ``<config_dir>/server.log``), and the console keeps the log as well:
Docker's ``docker logs`` and the desktop app's server log read stdout.
"""

import logging

import pytest

from pixlstash import app


@pytest.fixture
def root_logging():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    try:
        yield root
    finally:
        for handler in root.handlers:
            if handler not in handlers:
                handler.close()
        root.handlers = handlers
        root.setLevel(level)


@pytest.mark.parametrize("level", ["info", "debug", "warning"])
def test_a_configured_log_file_gets_the_log_at_every_level(
    tmp_path, root_logging, capsys, level
):
    log_file = tmp_path / "logs" / "server.log"
    app._configure_logging({"log_level": level, "log_file": str(log_file)})
    logging.getLogger("pixlstash.example").warning("example-line-%s", level)
    for handler in root_logging.handlers:
        handler.flush()

    assert f"example-line-{level}" in log_file.read_text(encoding="utf-8")
    console = capsys.readouterr()
    assert f"example-line-{level}" in console.out + console.err, (
        "the console lost the log"
    )


def test_the_default_level_writes_info_lines_to_the_file(tmp_path, root_logging):
    log_file = tmp_path / "server.log"
    app._configure_logging({"log_level": "info", "log_file": str(log_file)})
    logging.getLogger("pixlstash.example").info("example-info-line")
    for handler in root_logging.handlers:
        handler.flush()
    assert "example-info-line" in log_file.read_text(encoding="utf-8")


def test_a_null_log_file_logs_to_the_console_only(tmp_path, root_logging, capsys):
    app._configure_logging({"log_level": "info", "log_file": None})
    assert not any(isinstance(h, logging.FileHandler) for h in root_logging.handlers)
    logging.getLogger("pixlstash.example").info("example-console-line")
    console = capsys.readouterr()
    assert "example-console-line" in console.out + console.err
