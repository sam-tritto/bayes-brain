import os
from typing import Callable, List, Optional, Tuple, Union

from mcp.server.fastmcp import FastMCP

from bayes_brain.router import BayesianToolRouter
from bayes_brain.storage import SQLiteStorage


def create_mcp_server(
    server_name: str = "BayesBrain",
    db_path: str = "mcp_bandit.db",
    sub_tools: Optional[List[str]] = None,
    tool_executor: Optional[Callable[[str, str], Union[Tuple[str, bool], str]]] = None,
) -> FastMCP:
    """
    Configure and return a FastMCP server wrapping a BayesianToolRouter instance.

    Args:
        server_name: The display name of the FastMCP server.
        db_path: SQLite database path to store tool statistics.
        sub_tools: A list of candidate sub-tools the router can dynamically select.
        tool_executor: A callable taking (selected_tool, task_description) returning
                       either (output, success_bool) or just output (which defaults to success).
    """
    mcp = FastMCP(server_name)
    
    # Use SQLiteStorage for fast, persistent, local-cache statistics
    storage = SQLiteStorage(db_path)
    router = BayesianToolRouter(storage=storage)
    
    available_tools = sub_tools or ["local_pytest", "docker_sandbox", "fallback_api"]

    async def run_tool_logic(tool_name: str, task: str) -> Tuple[str, bool]:
        if tool_executor:
            import inspect
            if inspect.iscoroutinefunction(tool_executor):
                res = await tool_executor(tool_name, task)
            else:
                res = tool_executor(tool_name, task)

            if isinstance(res, tuple):
                return str(res[0]), bool(res[1])
            return str(res), True

        # Default fallback simulator for demonstrations
        if tool_name == "local_pytest":
            # Simulate failure on task requests with styling checks
            success = "style" not in task.lower()
            return f"Pytest execution: {'PASSED' if success else 'FAILED'}", success
        elif tool_name == "docker_sandbox":
            return "Docker sandbox execution completed successfully.", True
        else:
            return "Fallback API request dispatched and processed.", True

    @mcp.tool()
    async def execute_adaptive_action(task_description: str) -> str:
        """
        Dynamically routes task execution to the most reliable sub-tool.

        Args:
            task_description: A description of the code or integration task to execute.
        """
        # Thompson sampling selects the tool
        chosen_tool, trace_id = router.route_with_trace(
            context_text=task_description,
            candidate_tools=available_tools
        )

        try:
            result, success = await run_tool_logic(chosen_tool, task_description)
        except Exception as e:
            result, success = f"Adaptive execution encountered an error: {str(e)}", False

        # Submit execution feedback asynchronously
        router.feedback_by_trace(trace_id=trace_id, success=success)

        return f"Selected Tool: {chosen_tool}\nExecution Output:\n{result}"

    return mcp


if __name__ == "__main__":
    # Fetch DB path or configure defaults from environment
    db_path = os.environ.get("BAYES_DB_PATH", "mcp_bandit.db")
    server = create_mcp_server(db_path=db_path)
    server.run()
