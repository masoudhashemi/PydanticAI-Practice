from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, List

from pydantic_ai import Agent
from pydantic_ai.format_as_xml import format_as_xml
from pydantic_graph import BaseNode, Edge, End, Graph, GraphRunContext, HistoryStep


@dataclass
class ResearchState:
    research_question: str
    plan: str | None = None
    questions: List[str] = field(default_factory=list)
    search_results: dict[str, str] = field(default_factory=dict)
    followup_questions: List[str] = field(default_factory=list)
    summary: str | None = None
    agent_messages: list[dict] = field(default_factory=list)


# Define agents with specific roles
plan_agent = Agent(
    "openai:gpt-4o", result_type=str, system_prompt="Create a research plan to investigate the given question."
)

question_generator = Agent(
    "openai:gpt-4o", result_type=List[str], system_prompt="Generate specific questions to research the topic."
)

search_agent = Agent(
    "openai:gpt-4o",
    result_type=str,
    system_prompt="Simulate a search and provide relevant information for the question.",
)

followup_generator = Agent(
    "openai:gpt-4o", result_type=List[str], system_prompt="Generate follow-up questions based on the search results."
)

summarize_agent = Agent(
    "openai:gpt-4o", result_type=str, system_prompt="Synthesize all research findings into a comprehensive summary."
)


@dataclass
class Plan(BaseNode[ResearchState]):
    async def run(self, ctx: GraphRunContext[ResearchState]) -> GenerateQuestions:
        result = await plan_agent.run(f"Create a research plan for: {ctx.state.research_question}")
        ctx.state.plan = result.data
        print(f"Research Plan:\n{ctx.state.plan}\n")
        return GenerateQuestions()


@dataclass
class GenerateQuestions(BaseNode[ResearchState]):
    async def run(self, ctx: GraphRunContext[ResearchState]) -> Search:
        result = await question_generator.run(
            format_as_xml({"research_question": ctx.state.research_question, "plan": ctx.state.plan})
        )
        ctx.state.questions = result.data
        print("Generated Questions:")
        for q in ctx.state.questions:
            print(f"- {q}")
        return Search()


@dataclass
class Search(BaseNode[ResearchState]):
    async def run(self, ctx: GraphRunContext[ResearchState]) -> GenerateFollowup:
        for question in ctx.state.questions:
            result = await search_agent.run(question)
            ctx.state.search_results[question] = result.data
            print(f"\nSearching: {question}\nResults: {result.data[:200]}...")
        return GenerateFollowup()


@dataclass
class GenerateFollowup(BaseNode[ResearchState]):
    async def run(self, ctx: GraphRunContext[ResearchState]) -> SearchFollowup:
        result = await followup_generator.run(
            format_as_xml(
                {"original_question": ctx.state.research_question, "search_results": ctx.state.search_results}
            )
        )
        ctx.state.followup_questions = result.data
        print("\nFollow-up Questions:")
        for q in ctx.state.followup_questions:
            print(f"- {q}")
        return SearchFollowup()


@dataclass
class SearchFollowup(BaseNode[ResearchState]):
    async def run(self, ctx: GraphRunContext[ResearchState]) -> Summarize:
        for question in ctx.state.followup_questions:
            result = await search_agent.run(question)
            ctx.state.search_results[question] = result.data
            print(f"\nSearching followup: {question}\nResults: {result.data[:200]}...")
        return Summarize()


@dataclass
class Summarize(BaseNode[ResearchState]):
    async def run(self, ctx: GraphRunContext[ResearchState]) -> Annotated[End, Edge(label="complete")]:
        result = await summarize_agent.run(
            format_as_xml(
                {
                    "research_question": ctx.state.research_question,
                    "plan": ctx.state.plan,
                    "search_results": ctx.state.search_results,
                }
            )
        )
        ctx.state.summary = result.data
        print(f"\nFinal Summary:\n{ctx.state.summary}")
        return End(None)


# Create the research graph
research_graph = Graph(
    nodes=(Plan, GenerateQuestions, Search, GenerateFollowup, SearchFollowup, Summarize),
    state_type=ResearchState,
    run_end_type=None,
)


async def run_research(question: str, history_file: Path | None = None):
    """Run research with optional history saving/loading"""
    history: list[HistoryStep[ResearchState, None]] = []

    state = ResearchState(research_question=question)
    node = Plan()

    while True:
        node = await research_graph.next(node, history, state=state)
        if isinstance(node, End):
            break

    # Save history if file path provided
    if history_file:
        history_file.parent.mkdir(parents=True, exist_ok=True)
        # Convert bytes to string before writing
        history_data = research_graph.dump_history(history, indent=2)
        if isinstance(history_data, bytes):
            history_data = history_data.decode("utf-8")
        history_file.write_text(history_data)

    return state.summary


if __name__ == "__main__":
    import asyncio
    import sys

    research_question = "What are the environmental impacts of electric vehicles?"
    history_path = Path("research_graph_history.json")

    # Allow command line args to specify question and history file
    if len(sys.argv) > 1:
        research_question = sys.argv[1]
    if len(sys.argv) > 2:
        history_path = Path(sys.argv[2])

    summary = asyncio.run(run_research(research_question, history_path))
    print(f"\nFinal Summary:\n{summary}")
