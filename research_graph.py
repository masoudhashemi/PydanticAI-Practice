from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, List

from pydantic_ai import Agent
from pydantic_ai.format_as_xml import format_as_xml
from pydantic_graph import BaseNode, Edge, End, Graph, GraphRunContext, HistoryStep
from tavily import TavilyClient
import os
from dotenv import load_dotenv


@dataclass
class EvaluationResult:
    satisfactory: bool
    comment: str
    suggested_improvements: list[str]


# Load environment variables and initialize Tavily client
load_dotenv()
tavily_client = TavilyClient()

def format_sources(sources):
    """
    Formats a list of source dictionaries into a structured text for LLM input.
    """
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(sources, start=1):
        formatted_text += (
            f"Source {i}:\n"
            f"Title: {source['title']}\n"
            f"Url: {source['url']}\n"
            f"Content: {source['content']}\n\n"
        )
    return formatted_text.strip()


@dataclass
class ResearchState:
    research_question: str
    plan: str | None = None
    questions: List[str] = field(default_factory=list)
    search_results: dict[str, str] = field(default_factory=dict)
    followup_questions: List[str] = field(default_factory=list)
    summary: str | None = None
    agent_messages: list[dict] = field(default_factory=list)
    iteration_count: int = 0
    max_iterations: int = 3
    sources: List[dict] = field(default_factory=list)  # Add sources field


# Define agents with specific roles
plan_agent = Agent(
    "openai:gpt-4o", result_type=str, system_prompt="Create a research plan to investigate the given question."
)

question_generator = Agent(
    "openai:gpt-4o", result_type=List[str], system_prompt="Generate specific questions to research the topic."
)


followup_generator = Agent(
    "openai:gpt-4o", result_type=List[str], system_prompt="Generate follow-up questions based on the search results."
)

summarize_agent = Agent(
    "openai:gpt-4o", result_type=str, system_prompt="Synthesize all research findings into a comprehensive summary."
)

evaluate_agent = Agent(
    "openai:gpt-4o",
    result_type=EvaluationResult,
    system_prompt="Evaluate if the research results are satisfactory and suggest improvements if needed.",
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
            # Use Tavily for real web search
            search_results = tavily_client.search(
                question, 
                include_raw_content=False, 
                max_results=2
            )
            
            # Store the raw results and format them for the state
            ctx.state.sources.extend(search_results["results"])
            formatted_results = format_sources(search_results["results"])
            ctx.state.search_results[question] = formatted_results
            
            print(f"\nSearching: {question}")
            # Print first 200 chars of formatted results
            print(f"Results: {formatted_results[:200]}...")
            
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
            # Use Tavily for followup searches
            search_results = tavily_client.search(
                question, 
                include_raw_content=False, 
                max_results=2
            )
            
            # Store the raw results and format them for the state
            ctx.state.sources.extend(search_results["results"])
            formatted_results = format_sources(search_results["results"])
            ctx.state.search_results[question] = formatted_results
            
            print(f"\nSearching followup: {question}")
            print(f"Results: {formatted_results[:200]}...")
            
        return Summarize()


@dataclass
class Summarize(BaseNode[ResearchState]):
    async def run(self, ctx: GraphRunContext[ResearchState]) -> Evaluate:
        # Include all sources in the summary prompt
        all_sources = format_sources(ctx.state.sources)
        result = await summarize_agent.run(
            format_as_xml(
                {
                    "research_question": ctx.state.research_question,
                    "plan": ctx.state.plan,
                    "search_results": ctx.state.search_results,
                    "all_sources": all_sources,
                }
            )
        )
        ctx.state.summary = result.data
        print(f"\nFinal Summary:\n{ctx.state.summary}")
        return Evaluate()


@dataclass
class Evaluate(BaseNode[ResearchState]):
    async def run(
        self,
        ctx: GraphRunContext[ResearchState],
    ) -> Annotated[End, Edge(label="complete")] | GenerateQuestions:
        result = await evaluate_agent.run(
            format_as_xml(
                {
                    "research_question": ctx.state.research_question,
                    "summary": ctx.state.summary,
                    "search_results": ctx.state.search_results,
                }
            )
        )

        ctx.state.iteration_count += 1
        print(f"\nEvaluation (Iteration {ctx.state.iteration_count}/{ctx.state.max_iterations}):")
        print(f"Comment: {result.data.comment}")

        if result.data.satisfactory or ctx.state.iteration_count >= ctx.state.max_iterations:
            if ctx.state.iteration_count >= ctx.state.max_iterations:
                print("\nReached maximum iterations. Stopping research.")
            return End(None)
        else:
            print("\nSuggested improvements:")
            for improvement in result.data.suggested_improvements:
                print(f"- {improvement}")
            # Clear previous questions and results for new iteration
            ctx.state.questions = []
            ctx.state.followup_questions = []
            ctx.state.search_results = {}
            return GenerateQuestions()


# Create the research graph
research_graph = Graph(
    nodes=(Plan, GenerateQuestions, Search, GenerateFollowup, SearchFollowup, Summarize, Evaluate),
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
        # Add encoding parameter to handle Unicode characters
        history_file.write_text(history_data, encoding="utf-8")

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
