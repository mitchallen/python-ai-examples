"""The strategies, independent of any telemetry."""

from __future__ import annotations

import pytest

from ai_otel_109 import KeepAll, SlidingWindow, Summarizing, TokenBudget, estimate_tokens


def roles(decision):
    return [m["role"] for m in decision.messages]


def contents(decision):
    return [m["content"] for m in decision.messages]


# -- keep all ----------------------------------------------------------------


def test_keep_all_sends_everything(conversation_history):
    history = conversation_history(3)

    decision = KeepAll().prepare(history)

    assert decision.messages == history
    assert decision.dropped == 0


def test_keep_all_grows_with_every_turn(conversation_history):
    small = KeepAll().prepare(conversation_history(1))
    large = KeepAll().prepare(conversation_history(5))

    # The quadratic bill in one assertion: same questions, more to resend.
    assert len(large.messages) > len(small.messages)


# -- sliding window ----------------------------------------------------------


def test_sliding_window_keeps_the_last_n_exchanges(conversation_history):
    decision = SlidingWindow(max_turns=2).prepare(conversation_history(5))

    assert contents(decision)[1:] == [
        "question 3",
        "answer 3",
        "question 4",
        "answer 4",
    ]
    assert decision.dropped == 6


def test_sliding_window_never_drops_the_system_prompt(conversation_history):
    # Instructions are not history; losing them changes the model's behaviour.
    decision = SlidingWindow(max_turns=1).prepare(conversation_history(9))

    assert roles(decision)[0] == "system"


def test_a_short_conversation_is_untouched(conversation_history):
    decision = SlidingWindow(max_turns=5).prepare(conversation_history(2))

    assert decision.dropped == 0


# -- token budget ------------------------------------------------------------


def test_token_budget_trims_until_it_fits(conversation_history):
    history = conversation_history(8)

    decision = TokenBudget(max_tokens=20).prepare(history)

    assert decision.estimated_tokens <= 20
    assert decision.dropped > 0


def test_token_budget_keeps_the_most_recent(conversation_history):
    decision = TokenBudget(max_tokens=20).prepare(conversation_history(8))

    assert contents(decision)[-1] == "answer 7"


def test_token_budget_keeps_one_message_even_if_over_budget(conversation_history):
    # Sending nothing is not an option; the alternative to one oversized
    # message is no request at all.
    history = [{"role": "user", "content": "x" * 4000}]

    decision = TokenBudget(max_tokens=10).prepare(history)

    assert len(decision.messages) == 1


def test_estimate_counts_characters_not_words():
    assert estimate_tokens([{"role": "user", "content": "a" * 40}]) == 10


def test_estimate_tolerates_missing_content():
    assert estimate_tokens([{"role": "assistant"}]) == 0


# -- summarizing -------------------------------------------------------------


def test_no_compaction_before_the_trigger(conversation_history):
    calls = []
    strategy = Summarizing(lambda msgs: calls.append(msgs) or "summary", trigger_after=6)

    strategy.prepare(conversation_history(2))

    assert calls == []
    assert strategy.compactions == 0


def test_compaction_replaces_old_turns_with_a_summary(conversation_history):
    strategy = Summarizing(lambda msgs: "they discussed the rum", trigger_after=4)

    decision = strategy.prepare(conversation_history(5))

    assert strategy.compactions == 1
    assert decision.summarized == 6
    summary = [m for m in decision.messages if "Summary of the conversation" in m["content"]]
    assert summary and "they discussed the rum" in summary[0]["content"]
    # Recent turns survive verbatim.
    assert contents(decision)[-1] == "answer 4"


def test_the_summary_is_a_system_message(conversation_history):
    # Not a user turn: it is context about the conversation, and labelling it
    # otherwise invites the model to answer it.
    strategy = Summarizing(lambda msgs: "summary", trigger_after=4)

    decision = strategy.prepare(conversation_history(5))

    summary = next(m for m in decision.messages if "Summary" in m["content"])
    assert summary["role"] == "system"


def test_it_does_not_recompact_every_turn(conversation_history):
    # The first implementation did, which measured worse than keeping
    # everything: a model call per turn to save a few hundred characters.
    strategy = Summarizing(lambda msgs: "summary", trigger_after=4, recompact_after=4)

    strategy.prepare(conversation_history(5))
    strategy.prepare(conversation_history(6))

    assert strategy.compactions == 1


def test_it_recompacts_once_enough_has_aged_out(conversation_history):
    strategy = Summarizing(lambda msgs: "summary", trigger_after=4, recompact_after=4)

    strategy.prepare(conversation_history(5))
    strategy.prepare(conversation_history(8))

    assert strategy.compactions == 2


def test_messages_awaiting_compaction_are_still_sent(conversation_history):
    # Aged out but not yet summarised: they ride along verbatim, so nothing is
    # ever dropped without first being summarised.
    strategy = Summarizing(lambda msgs: "summary", trigger_after=4, recompact_after=6)

    strategy.prepare(conversation_history(5))
    decision = strategy.prepare(conversation_history(6))

    assert strategy.compactions == 1
    sent = contents(decision)
    # Aged out since the compaction, not yet summarised -> sent verbatim.
    assert "question 3" in sent
    # Already covered by the summary -> not resent.
    assert "question 1" not in sent


def test_strategies_report_their_own_name():
    assert KeepAll().name == "keep_all"
    assert SlidingWindow().name == "sliding_window"
    assert TokenBudget().name == "token_budget"
    assert Summarizing(lambda m: "").name == "summarizing"
