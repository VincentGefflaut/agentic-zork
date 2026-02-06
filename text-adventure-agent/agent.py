"""
Student Agent for Text Adventure Games

This is your submission file. Implement the StudentAgent class to play
text adventure games using the MCP server you also implement.

Your agent should:
1. Connect to the MCP server via the provided client
2. Use the ReAct pattern (Thought -> Action -> Observation)
3. Call MCP tools to interact with the game
4. Maximize the game score within the step limit

Required method:
    async def run(self, client, game, max_steps, seed, verbose) -> RunResult

The 'client' is a FastMCP Client already connected to your MCP server.
Use it to call tools like: await client.call_tool("play_action", {"action": "look"})

Tips:
- Start by looking around and understanding your environment
- Keep track of visited locations to avoid loops
- Pick up useful items (lamp, sword, etc.)
- The seed parameter should be used to set your LLM's seed for reproducibility
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# =============================================================================
# LLM Configuration - DO NOT MODIFY
# =============================================================================

# Model to use (fixed for fair evaluation)
LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# Initialize the LLM client (uses HF_TOKEN from environment)
_hf_token = os.getenv("HF_TOKEN")
if not _hf_token:
    raise ValueError("HF_TOKEN not found. Set it in your .env file.")

LLM_CLIENT = InferenceClient(token=_hf_token)


def call_llm(prompt: str, system_prompt: str, seed: int, max_tokens: int = 300) -> str:
    """
    Call the LLM with the given prompt. Use this function in your agent.
    
    Args:
        prompt: The user prompt (current game state, history, etc.)
        system_prompt: The system prompt (instructions for the agent)
        seed: Random seed for reproducibility
        max_tokens: Maximum tokens in response (default: 300)
        
    Returns:
        The LLM's response text
        
    Example:
        response = call_llm(
            prompt="You are in a forest. What do you do?",
            system_prompt=SYSTEM_PROMPT,
            seed=42,
        )
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    
    response = LLM_CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.0,  # Deterministic for reproducibility
        max_tokens=max_tokens,
        seed=seed,
    )
    
    return response.choices[0].message.content


@dataclass
class RunResult:
    """Result of running the agent. Do not modify this class."""
    final_score: int
    max_score: int
    moves: int
    locations_visited: set[str]
    game_completed: bool
    error: Optional[str] = None
    history: list[tuple[str, str, str]] = field(default_factory=list)


# =============================================================================
# System Prompt - Customize this for your agent
# =============================================================================

SYSTEM_PROMPT = """You are an expert text adventure game player. Your goal is to explore, collect treasures, and maximize your score.

AVAILABLE TOOLS (use these via MCP):
1. play_action - Execute game commands (north, take lamp, open mailbox, etc.)
2. memory - Get current game state, score, and recent history
3. get_map - See explored locations and connections
4. inventory - Check what you're carrying

VALID GAME COMMANDS for play_action:
- Movement: north, south, east, west, up, down, enter, exit
- Objects: take <item>, drop <item>, open <thing>, close <thing>, examine <thing>
- Light: turn on lamp, turn off lamp
- Combat: attack <enemy> with <weapon>
- Other: inventory, look, read <thing>, wait

FORBIDDEN (will NOT work): check, inspect, search, grab, use, help

RESPOND IN THIS EXACT FORMAT (no markdown):
THOUGHT: <brief reasoning about what to do next>
TOOL: <tool_name>
ARGS: <JSON arguments>

Examples:
THOUGHT: I need to see what's around me.
TOOL: play_action
ARGS: {"action": "look"}

THOUGHT: Let me check my current state and score.
TOOL: memory
ARGS: {}

THOUGHT: The mailbox might contain something useful.
TOOL: play_action
ARGS: {"action": "open mailbox"}

STRATEGY:
1. Start by looking around and checking memory
2. Explore systematically - try all directions
3. Pick up useful items (lamp, sword, etc.)
4. Open containers (mailbox, window, etc.)
5. Use get_map to avoid getting lost
6. Turn on lamp before dark areas!

DO NOT repeat the same action multiple times in a row."""


# =============================================================================
# Student Agent - IMPLEMENT THIS CLASS
# =============================================================================

class StudentAgent:
    """
    Your ReAct agent implementation.
    
    TODO:
    1. Implement the run() method with the ReAct loop
    2. Parse LLM responses to extract tool calls
    3. Track state and avoid loops
    
    Use the provided call_llm() function to interact with the LLM.
    """
    
    def __init__(self):
        """Initialize your agent here."""
        # TODO: Initialize any state tracking you need
        # self.history = []
        # self.visited_locations = set()
        self.history: list[dict] = []
        self.recent_actions: list[str] = []
        self.score: int = 0
    
    async def run(
        self,
        client,  # FastMCP Client connected to your MCP server
        game: str,
        max_steps: int,
        seed: int,
        verbose: bool = False,
    ) -> RunResult:
        """
        Run the agent for a game session.
        
        Args:
            client: FastMCP Client connected to your MCP server
            game: Name of the game being played (e.g., "zork1")
            max_steps: Maximum number of steps to take
            seed: Random seed for reproducibility (use for LLM calls)
            verbose: Whether to print detailed output
            
        Returns:
            RunResult with final score and statistics
        """
        # TODO: Implement your ReAct loop here
        #
        # Basic structure:
        # 1. Get initial observation (call play_action with "look")
        # 2. Loop for max_steps:
        #    a. Build prompt with current observation and history
        #    b. Call LLM to get thought and action
        #    c. Parse the response to extract tool and args
        #    d. Call the tool via client.call_tool(tool_name, args)
        #    e. Update history and state
        #    f. Check for game over
        # 3. Return RunResult with final statistics
        
        # Example of calling a tool:
        # result = await client.call_tool("play_action", {"action": "look"})
        # observation = result[0].text if result else "No response"
        
        # Example of calling the LLM:
        # response = call_llm(
        #     prompt="Current observation: " + observation,
        #     system_prompt=SYSTEM_PROMPT,
        #     seed=seed,
        # )
        
        # Placeholder implementation - replace with your code
        locations_visited = set()
        history = []
        moves = 0

        # TODO: Your implementation here
        # ...
        # Get list of available tools
        tools = await client.list_tools()
        tool_names = [t.name for t in tools]
        
        # Get initial observation
        result = await client.call_tool("play_action", {"action": "look"})
        observation = self._extract_result(result)
        
        # Track initial location
        location = observation.split("\n")[0] if observation else "Unknown"
        locations_visited.add(location)
        
        if verbose:
            print(f"\n{observation}")
        
        # Main ReAct loop
        for step in range(1, max_steps + 1):
            # Build prompt with context
            prompt = self._build_prompt(observation)
            if verbose:
                print(f"PROMPT:\n{prompt}")
            
            # Call LLM for reasoning (use step-based seed for variety)
            response = call_llm(prompt, SYSTEM_PROMPT, seed + step)
            
            # Parse the response
            thought, tool_name, tool_args = self._parse_response(response, tool_names)
            
            if verbose:
                print(f"\n--- Step {step} ---")
                print(f"[THOUGHT] {thought}")
                print(f"[TOOL] {tool_name}({tool_args})")
            
            # Validate and fix common issues
            tool_name, tool_args = self._validate_tool_call(tool_name, tool_args, tool_names)
            
            # Loop detection
            if tool_name == "play_action":
                action = tool_args.get("action", "look")
                self.recent_actions.append(action)
                if len(self.recent_actions) > 5:
                    self.recent_actions = self.recent_actions[-5:]
                
                # Detect loops - if same action 3 times, force "look"
                if len(self.recent_actions) >= 3 and len(set(self.recent_actions[-3:])) == 1:
                    if verbose:
                        print(f"[WARNING] Loop detected - forcing 'look'")
                    tool_args = {"action": "look"}
                    self.recent_actions.append("look")
                
                moves += 1
            
            # Execute the tool
            try:
                result = await client.call_tool(tool_name, tool_args)
                observation = self._extract_result(result)
                
                if verbose:
                    print(f"[RESULT] {observation[:200]}...")
            except Exception as e:
                observation = f"Error: {e}"
                if verbose:
                    print(f"[ERROR] {e}")
            
            # Track location
            location = observation.split("\n")[0] if observation else "Unknown"
            locations_visited.add(location)
            
            # Update history
            self.history.append({
                "step": step,
                "thought": thought,
                "tool": tool_name,
                "args": tool_args,
                "result": observation[:200]
            })
            if len(self.history) > 10:
                self.history = self.history[-10:]
            
            # Track score from observation
            self._update_score(observation)
            
            # Record in result history
            history.append((thought, f"{tool_name}({tool_args})", observation[:100]))
            
            # Check for game over
            if self._is_game_over(observation):
                if verbose:
                    print("\n*** GAME OVER ***")
                break
        
        return RunResult(
            final_score=self.score,
            max_score=350,
            moves=moves,
            locations_visited=locations_visited,
            game_completed=self._is_game_over(observation),
            history=history,
        )
    
    def _build_prompt(self, observation: str) -> str:
        """Build the prompt for the LLM with context."""
        parts = []
        
        parts.append(f"Current Score: {self.score}")
        
        # Recent history
        if self.history:
            if len(self.history) > 1:
                parts.append("Actions history:")
                for entry in self.history[-10:-1]:
                    parts.append(f"  > {entry['tool']}({entry['args']}) -> {entry['result'][:100]}{"..." if len(entry['result']) > 100 else ""}")
            parts.append("Last action:")
            parts.append(f"  > {self.history[-1]['tool']}({self.history[-1]['args']}) -> {observation}")

            # Warn about repeated actions
            if len(self.recent_actions) >= 3 and len(set(self.recent_actions[-3:])) == 1:
                parts.append(f"\n[WARNING: You've been doing '{self.recent_actions[-1]}' repeatedly. TRY SOMETHING DIFFERENT!]")

        parts.append("What do you do next?")
        
        return "\n".join(parts)
    
    def _parse_response(self, response: str, valid_tools: list[str]) -> tuple[str, str, dict]:
        """Parse the LLM response to extract thought, tool, and arguments."""
        thought = "No reasoning provided"
        tool_name = "play_action"
        tool_args = {"action": "look"}
        
        lines = response.strip().split("\n")
        
        for line in lines:
            line_clean = line.strip()
            line_upper = line_clean.upper()
            
            if line_upper.startswith("THOUGHT:"):
                thought = line_clean.split(":", 1)[1].strip()
            
            elif line_upper.startswith("TOOL:"):
                raw_tool = line_clean.split(":", 1)[1].strip().lower()
                raw_tool = raw_tool.replace("**", "").replace("*", "").replace("`", "")
                raw_tool = raw_tool.split()[0] if raw_tool else "play_action"
                tool_name = raw_tool
            
            elif line_upper.startswith("ARGS:"):
                args_part = line_clean.split(":", 1)[1].strip()
                try:
                    args_part = args_part.replace("'", '"')
                    tool_args = json.loads(args_part)
                except json.JSONDecodeError:
                    match = re.search(r'"action"\s*:\s*"([^"]+)"', args_part)
                    if match:
                        tool_args = {"action": match.group(1)}
                    else:
                        tool_args = {"action": "look"}
        
        return thought, tool_name, tool_args
    
    def _validate_tool_call(self, tool_name: str, tool_args: dict, valid_tools: list[str]) -> tuple[str, dict]:
        """Validate and fix common tool call issues."""
        # Fix tool name
        if tool_name not in valid_tools:
            if tool_name in ["action", "do", "command"]:
                tool_name = "play_action"
            elif tool_name in ["map", "location"]:
                tool_name = "get_map"
            elif tool_name in ["mem", "state", "status"]:
                tool_name = "memory"
            elif tool_name in ["inv", "items"]:
                tool_name = "inventory"
            else:
                tool_name = "play_action"
        
        # Fix action verbs
        if tool_name == "play_action":
            action = tool_args.get("action", "look")
            
            invalid_verb_map = {
                "check": "examine",
                "inspect": "examine",
                "search": "look",
                "grab": "take",
                "pick": "take",
                "use": "examine",
                "investigate": "examine",
            }
            
            words = action.lower().split()
            if words and words[0] in invalid_verb_map:
                words[0] = invalid_verb_map[words[0]]
                action = " ".join(words)
            
            action = action.lower().strip()
            action = action.replace("**", "").replace("*", "").replace("`", "")
            action = " ".join(action.split())
            
            tool_args["action"] = action
        
        return tool_name, tool_args
    
    def _extract_result(self, result) -> str:
        """Extract text from MCP tool result."""
        if hasattr(result, 'content') and result.content:
            return result.content[0].text
        if isinstance(result, list) and result:
            return result[0].text if hasattr(result[0], 'text') else str(result[0])
        return str(result)
    
    def _update_score(self, text: str) -> None:
        """Update score from game text."""
        patterns = [
            r'Score:\s*(\d+)',
            r'score[:\s]+(\d+)',
            r'\[Score:\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                self.score = max(self.score, int(match.group(1)))
    
    def _is_game_over(self, text: str) -> bool:
        """Check if the game is over."""
        game_over_phrases = [
            "game over",
            "you have died",
            "you are dead",
            "*** you have died ***",
        ]
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in game_over_phrases)


# =============================================================================
# For local testing
# =============================================================================

async def test_agent():
    """Test the agent locally."""
    from fastmcp import Client
    
    # Path to your MCP server
    server_path = "mcp_server.py"
    
    agent = StudentAgent()
    
    async with Client(server_path) as client:
        result = await agent.run(
            client=client,
            game="zork1",
            max_steps=10,
            seed=42,
            verbose=True,
        )
        
        print(f"\nFinal Score: {result.final_score}")
        print(f"Moves: {result.moves}")
        print(f"Locations: {len(result.locations_visited)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agent())
