"""A configured log file and the console both receive the log.

Every install's server-config carries ``log_file``, and the console must keep
the log as well: ``docker logs`` and the desktop app's server log read stdout.
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


@pytest.mark.parametrize("level", ["info", "debug"])
def test_the_log_file_and_the_console_both_get_the_log(
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
