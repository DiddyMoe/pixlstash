import subprocess

import pytest

from pixlstash.utils import system_utils, vram_utils


@pytest.fixture(autouse=True)
def a_card_host(monkeypatch):
    """Simulate a non-Apple host for every test in this module.

    These tests describe the **card** path, and they say so by patching
    ``nvidia-smi``. That was enough while "has a GPU" meant "has CUDA"; it is
    not enough now that ``query_total_vram_mb`` and ``default_max_vram_gb``
    both branch on the memory model first, because on an Apple machine they
    answer from Metal and never reach the mocked ``nvidia-smi`` at all. The
    tests then passed on CI's Linux runners and failed on a developer's Mac -
    a test whose result depends on the machine running it, which is the thing
    the simulated-host discipline in ``test_mps_memory_and_providers.py``
    exists to avoid. The unified-memory answers are asserted there.
    """
    monkeypatch.setattr(vram_utils, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(system_utils, "is_apple_silicon", lambda: False)
    # The card total is kept once read: each test's simulated card starts
    # unread, and monkeypatch puts back whatever the process had read before.
    monkeypatch.setattr(vram_utils, "_card_total_mb", 0)


@pytest.mark.parametrize(
    "total_mb, expected_gb",
    [("32768", 16.0), ("12288", 6.0), ("8192", 4.0)],
)
def test_default_max_vram_gb_is_card_aware(monkeypatch, total_mb, expected_gb):
    monkeypatch.setattr(
        subprocess, "check_output", lambda *args, **kwargs: f"{total_mb}\n"
    )
    assert system_utils.default_max_vram_gb() == expected_gb


def test_default_max_vram_gb_falls_back_to_6_without_nvidia_smi(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "check_output", missing)
    assert system_utils.default_max_vram_gb() == 6.0


def test_default_max_vram_gb_falls_back_to_6_when_total_is_zero(monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "0\n")
    assert system_utils.default_max_vram_gb() == 6.0


@pytest.mark.parametrize(
    "total_mb, expected_gb",
    [("32768", 32.0), ("12288", 12.0), ("8192", 8.0)],
)
def test_the_settable_ceiling_is_the_card(monkeypatch, total_mb, expected_gb):
    """The default must always be settable: 16 GB on a 32 GB card was not."""
    monkeypatch.setattr(
        subprocess, "check_output", lambda *args, **kwargs: f"{total_mb}\n"
    )
    assert system_utils.max_vram_budget_gb() == expected_gb
    assert system_utils.default_max_vram_gb() <= system_utils.max_vram_budget_gb()


def test_the_settable_ceiling_falls_back_without_a_card(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "check_output", missing)
    assert system_utils.max_vram_budget_gb() == system_utils.MAX_VRAM_BUDGET_GB


def test_the_card_is_asked_once(monkeypatch):
    """``GET /users/me/config`` reads the card on every request; one subprocess."""
    calls = []

    def nvidia_smi(*args, **kwargs):
        calls.append(args)
        return "16384\n"

    monkeypatch.setattr(subprocess, "check_output", nvidia_smi)
    assert vram_utils.query_total_vram_mb() == 16384
    assert vram_utils.query_total_vram_mb() == 16384
    assert len(calls) == 1


def test_a_card_that_has_not_answered_is_asked_again(monkeypatch):
    """A 0 can be a driver still starting; keeping it would pin the slider."""
    answers = iter([FileNotFoundError("nvidia-smi"), "16384\n"])

    def nvidia_smi(*args, **kwargs):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(subprocess, "check_output", nvidia_smi)
    assert vram_utils.query_total_vram_mb() == 0
    assert vram_utils.query_total_vram_mb() == 16384


def test_a_card_that_does_not_answer_in_time_is_logged_and_asked_again(
    monkeypatch, caplog
):
    """A wedged driver must not hold the settings request open."""
    timeouts = []
    answers = iter(
        [
            subprocess.TimeoutExpired("nvidia-smi", vram_utils.NVIDIA_SMI_TIMEOUT_S),
            "16384\n",
        ]
    )

    def nvidia_smi(*args, **kwargs):
        timeouts.append(kwargs.get("timeout"))
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(subprocess, "check_output", nvidia_smi)
    with caplog.at_level("WARNING"):
        assert vram_utils.query_total_vram_mb() == 0
    assert "nvidia-smi did not report" in caplog.text
    assert vram_utils.query_total_vram_mb() == 16384
    assert timeouts == [vram_utils.NVIDIA_SMI_TIMEOUT_S] * 2
