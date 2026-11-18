"""Payment Planner Agent assembly."""

from __future__ import annotations

from agents import Agent, ModelSettings, RunContextWrapper, function_tool

if __package__:
    from .instructions import instructions_with_profile
    from .long_term_memory import build_recalled_memory_block
    from .memory import PaymentPlannerContext, planner_profile_summary, save_planner_preference
    from .tools import TRANSACTION_TOOLS
    from .semantic_search import classify_transaction, search_similar_transactions as find_similar_transactions
else:
    from instructions import instructions_with_profile
    from long_term_memory import build_recalled_memory_block
    from memory import (
        PaymentPlannerContext,
        planner_profile_summary,
        save_planner_preference,
    )
    from tools import TRANSACTION_TOOLS
    from semantic_search import (
        classify_transaction,
        search_similar_transactions as find_similar_transactions,
    )


def dynamic_instructions(
    wrapper: RunContextWrapper[PaymentPlannerContext],
    _agent: Agent[PaymentPlannerContext],
) -> str:
    """Inject durable profile facts into every model run."""

    return instructions_with_profile(planner_profile_summary(wrapper.context))


@function_tool
def classify_payment_transaction(user_input: str) -> dict:
    """Classify one transaction by parser, precise rules, then semantic retrieval.

    Args:
        user_input: One raw transaction such as ``CITY RAIL DAY TICKET $8.00``.
    """

    return classify_transaction(user_input)


@function_tool
def search_similar_transactions(user_input: str) -> dict:
    """Find verified semantic neighbors when deterministic categorization is Other.

    Args:
        user_input: One raw transaction such as ``CITY RAIL DAY TICKET $8.00``.
    """

    return find_similar_transactions(user_input)


def build_agent(recalled: list[dict] | None = None, *, model: str | None = None,
                max_output_tokens: int | None = None) -> Agent[PaymentPlannerContext]:
    """Construct the Payment Planner with transaction and memory tools."""

    recalled_block = build_recalled_memory_block(recalled or [])

    def instructions_for_run(
        wrapper: RunContextWrapper[PaymentPlannerContext],
        _agent: Agent[PaymentPlannerContext],
    ) -> str:
        return instructions_with_profile(
            planner_profile_summary(wrapper.context),
            recalled_block,
        )

    return Agent[PaymentPlannerContext](
        name="Payment Planner",
        model=model,
        model_settings=ModelSettings(max_tokens=max_output_tokens),
        instructions=instructions_for_run,
        tools=[
            *TRANSACTION_TOOLS,
            classify_payment_transaction,
            search_similar_transactions,
            save_planner_preference,
        ],
    )
