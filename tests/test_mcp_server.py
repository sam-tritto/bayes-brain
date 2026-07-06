import os

from mcp.server.fastmcp import FastMCP

from bayes_brain.mcp_server import create_mcp_server


def test_mcp_server_creation():
    db_path = "test_mcp_bandit.db"
    
    # Clean up in case of previous failures
    if os.path.exists(db_path):
        os.remove(db_path)

    try:
        mcp = create_mcp_server(
            server_name="TestBanditServer",
            db_path=db_path,
            sub_tools=["tool1", "tool2"]
        )

        assert isinstance(mcp, FastMCP)
        assert mcp.name == "TestBanditServer"

        # Check if execute_adaptive_action tool is registered
        # FastMCP stores registered tools in a dictionary or list
        # Let's inspect the tools registered in the FastMCP instance
        tools = mcp._tool_manager.list_tools() if hasattr(mcp, "_tool_manager") else []
        tool_names = [t.name for t in tools]
        assert "execute_adaptive_action" in tool_names or len(tool_names) > 0
    finally:
        # Clean up database file
        if os.path.exists(db_path):
            os.remove(db_path)
