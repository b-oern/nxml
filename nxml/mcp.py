"""
  pip install mcp
"""
from mcp.server.fastmcp import FastMCP

# Create an MCP server
print("Starting MCP")
mcp = FastMCP("Demo")
# add_tool(fn: AnyFunction,name: str description: str ,annotations: ToolAnnotations  )

def log(message):
    try:
        with open('/home/pi/mcp.log', 'a') as f:
            f.write(str(message) + "\n")
    except:
        pass

#@mcp.tool(description="A simple echo tool")
@mcp.tool()
def switch(state: bool) -> bool:
    """ Turn the switch on or off,
      return bool for sucess of the operation"""
    log("ToolCall switch("+str(state)+")")
    return True


# Add a dynamic greeting resource
#@mcp.resource("greeting://{name}")
#def get_greeting(name: str) -> str:
#    """Get a personalized greeting"""
#    return f"Hello, {name}!"


if __name__ == "__main__":
    log("NX-MCP Serv")
    mcp.run()
