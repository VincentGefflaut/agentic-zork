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
THINK_MODE = False
SUMMARY_MODE = False
SUMMARIZE_RESULTS = False
NOTEPAD = True

# Agent Configuration
MAX_HISTORY_LENGTH = 20  # Number of recent actions to include in the prompt

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

SYSTEM_PROMPT = f"""You are an expert text adventure game player. Your goal is to explore, collect treasures, and maximize your score.

RESPOND IN THIS EXACT FORMAT (no markdown):{"\nTHOUGHT: <brief reasoning about what to do next>" if THINK_MODE else ""}{"\nRESULT_SUMMARY: <summary of the last action output>" if SUMMARIZE_RESULTS else ""}
TOOL: <tool_name>
ARGS: <JSON arguments>

TOOLS USAGE:
1. play_action(action: str) - Execute game commands (north, take lamp, open mailbox, etc.)
2. memory() - Get current game state, score, and recent history
3. get_map() - See explored locations and connections
4. inventory() - Check what you're carrying
5. get_valid_actions() - Get a list of likely valid actions from the current location.
{"6. append_notepad(note: str) - Append a note to your persistent notepad." if NOTEPAD else ""}
{"7. replace_in_notepad(old_string: str, new_string: str) - Edit an existing part of your persistent notepad." if NOTEPAD else ""}
{"6. append_summary(summary: str) - Add text to the existing summary of past actions to help you remember. ONLY call this tool when asked." if SUMMARY_MODE else ""}

VALID GAME COMMANDS for play_action:
- Movement: north, south, east, west, up, down, enter, exit
- Objects: take <item>, drop <item>, open <thing>, close <thing>, examine <thing>
- Light: turn on lamp, turn off lamp
- Combat: attack <enemy> with <weapon>
- Other: inventory, look, read <thing>, wait

FORBIDDEN (will NOT work): check, inspect, search, grab, use, help

Examples:{"\nTHOUGHT: I need to see what's around me." if THINK_MODE else ""}{"\nRESULT_SUMMARY: Starting point, score 0." if SUMMARIZE_RESULTS else ""}
TOOL: play_action
ARGS: {{"action": "look"}}
{"\nTHOUGHT: Let me check my current state and score." if THINK_MODE else ""}{"\nRESULT_SUMMARY: The leaflet says nothing interesting." if SUMMARIZE_RESULTS else ""}
TOOL: memory
ARGS: {{}}
{"\nTHOUGHT: The mailbox might contain something useful." if THINK_MODE else ""}{"\nRESULT_SUMMARY: There is a fountain." if SUMMARIZE_RESULTS else ""}
TOOL: play_action
ARGS: {{"action": "open mailbox"}}

STRATEGY:
1. Start by looking around and checking memory
2. Explore systematically - try all directions
3. Pick up useful items (lamp, sword, etc.)
4. Open containers (mailbox, window, etc.)
5. Use get_map to avoid getting lost
6. Turn on lamp before dark areas!

{"Actively keep the notepad updated with essential information: major achievements (score gains, treasures, unlocked paths) and failures (dead ends, dangerous actions, blocked routes, unsuccessful actions)." if NOTEPAD else ""}

{"""Every few steps, you'll be asked to summarize your visible actions history using tool append_summary. Be as concise as possible. ONLY call this tool when asked.
Example summary: took key from mailbox at starting point. Mansion north starting point. Opened chest with key: +1 point.""" if SUMMARY_MODE else ""}
{"""When giving RESULT SUMMARY, focus on changes in score, new locations discovered, and important items found. Only summarize the result of last action taken. This helps you keep track of progress without overwhelming you with details.""" if SUMMARIZE_RESULTS else ""}

DO NOT repeat the same action multiple times in a row. Only call one tool per step."""


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
        self.summary: list[str] = []
        self.notepad: str = ""
        self.locations_visited: set[str] = set()
        self.movement_feedback: str = ""
        self.place_visit_steps: dict[str, list[int]] = {}
        self.place_last_session: dict[str, list[tuple[str, str]]] = {}
        self.current_location: str = ""
        self.current_place_actions: list[tuple[str, str]] = []
    
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
        self.notepad = ""
        self.locations_visited = set()
        self.movement_feedback = ""
        self.place_visit_steps = {}
        self.place_last_session = {}
        self.current_location = ""
        self.current_place_actions = []

        # TODO: Your implementation here
        
        # Add game name to system prompt hoping the LLM has seen it before
        global SYSTEM_PROMPT
        SYSTEM_PROMPT += f"\n\nYou are playing: {game.upper()}"

        # Get list of available tools
        tools = await client.list_tools()
        tool_names = [t.name for t in tools]
        if verbose:
            print(f"[AVAILABLE TOOLS]: {tool_names}")
        
        # Get initial observation
        result = await client.call_tool("play_action", {"action": "look"})
        observation = self._extract_result(result)
        
        # Track initial location
        location = self._extract_location(observation)
        locations_visited.add(location)
        self.locations_visited.add(location)
        self.current_location = location
        self.place_visit_steps[location] = [0]
        
        if verbose:
            print(f"[SYSTEM PROMPT]:\n{SYSTEM_PROMPT}\n")
        
        # Main ReAct loop
        for step in range(1, max_steps + 1):
            # Get possible moves at this point
            # try:
            #     valid_actions_result = await client.call_tool("get_valid_actions", {})
            #     valid_actions = self._extract_result(valid_actions_result).split(", ") if valid_actions_result else []
            # except Exception as e: 
            #     valid_actions = []
            #     if verbose: print(f"[ERROR getting valid actions]: {e}")

            # Build prompt with context
            prompt = self._build_prompt(observation)
            self.movement_feedback = ""
            
            # Call LLM for reasoning (use step-based seed for variety)
            response = call_llm(prompt, SYSTEM_PROMPT, seed + step)

            # Parse the response
            thought, result_summary, tool_name, tool_args = self._parse_response(response, tool_names)
            
            if verbose:
                print(f"\n--- Step {step} ---")
                print(f"[PROMPT]:\n{prompt}")
                print(f"\n[RAW RESPONSE]:\n{response}")
                # if THINK_MODE:
                #     print(f"\n[THOUGHT] {thought}")
                # if SUMMARIZE_RESULTS:
                #     print(f"\n[RESULT SUMMARY] {result_summary}")
                # print(f"\n[TOOL] {tool_name}({tool_args})")
            
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
            elif tool_name == "append_summary":
                summary = tool_args.get("summary", "")
                self.summary.append(summary)
                # Erase the part of history that was summarized
                self.history = self.history[-1:]
                if verbose:
                    print(f"[SUMMARY APPENDED] {summary}")
                continue  # Don't call a tool for summary updates
            
            # Execute the tool
            try:
                result = await client.call_tool(tool_name, tool_args)
                observation = self._extract_result(result)

                if NOTEPAD and tool_name in ["append_notepad", "replace_in_notepad"]:
                    self.notepad = observation

                if tool_name == "play_action":
                    action_taken = tool_args.get("action", "")
                    old_location = self.current_location
                    new_location = self._extract_location(observation, fallback=old_location)

                    self.current_place_actions.append((action_taken, observation[:300]))

                    if self._is_movement_action(action_taken) and new_location != old_location:
                        if old_location:
                            self.place_last_session[old_location] = list(self.current_place_actions)

                        previous_visits = self.place_visit_steps.get(new_location, [])
                        if not previous_visits:
                            self.movement_feedback = (
                                f"[Navigation] Great discovery: '{new_location}' is a new place."
                            )
                        else:
                            steps_since = step - previous_visits[-1]
                            feedback_lines = [
                                f"[Navigation] You are back at '{new_location}', last seen {steps_since} steps ago."
                            ]

                            if steps_since > MAX_HISTORY_LENGTH:
                                recap = self.place_last_session.get(new_location, [])
                                if recap:
                                    feedback_lines.append(
                                        f"Because this was more than {MAX_HISTORY_LENGTH} steps ago, here is a recap of prior actions at this place:"
                                    )
                                    for old_action, old_result in recap:
                                        feedback_lines.append(
                                            f"  - {old_action} -> {old_result}"
                                        )
                            self.movement_feedback = "\n".join(feedback_lines)

                        self.current_location = new_location
                        self.current_place_actions = []
                        self.place_visit_steps.setdefault(new_location, []).append(step)
                        locations_visited.add(new_location)
                        self.locations_visited.add(new_location)
                
                if verbose:
                    print(f"[RESULT] {observation}")
            except Exception as e:
                observation = f"Error: {e}"
                if verbose:
                    print(f"[ERROR] {e}")
            
            # Track location
            location = self._extract_location(observation, fallback=self.current_location)
            locations_visited.add(location)
            self.locations_visited.add(location)
            
            # Update history
            self.history.append({
                "step": step,
                "thought": thought,
                "tool": tool_name,
                "args": tool_args,
                "result": observation[:1000],
                "result_summary": result_summary
            })
            
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
        parts.append(f"Places Visited: {len(self.locations_visited)}")

        if self.movement_feedback:
            parts.append(self.movement_feedback)

        if NOTEPAD:
            parts.append("Notepad:")
            parts.append(self.notepad if self.notepad else "(empty)")

        # Summary
        if SUMMARY_MODE and self.summary:
            parts.append("Summary of past hidden actions:")
            parts.extend(self.summary)
        
        # Recent history
        result_key = "result_summary" if SUMMARIZE_RESULTS else "result"
        if self.history:
            if len(self.history) > 1:
                parts.append("Actions history:")
                for entry in self.history[-MAX_HISTORY_LENGTH:-1]:
                    parts.append(f"  > {entry['tool']}({entry['args']}) -> {entry[result_key][:1000]}{"..." if len(entry[result_key]) > 1000 else ""}")
            parts.append("Last action:")
            parts.append(f"  > {self.history[-1]['tool']}({self.history[-1]['args']}) -> {observation}")

            # Warn about repeated actions
            if len(self.recent_actions) >= 3 and len(set(self.recent_actions[-3:])) == 1:
                parts.append(f"\n[WARNING: You've been doing '{self.recent_actions[-1]}' repeatedly. TRY SOMETHING DIFFERENT!]")

        if not self.history:
            parts.append(f"Initial observation:\n{observation}")
        # if valid_actions:
        #     parts.append("Non-exhaustive list of valid actions: " + ", ".join(valid_actions[:10]))
        if SUMMARY_MODE and self.history and len(self.history) % MAX_HISTORY_LENGTH == 0:
            if not self.summary:
                parts.append(f"\nCall the tool append_summary to create a summary of the last {MAX_HISTORY_LENGTH} actions to help you remember.")
            else:
                parts.append(f"Call the tool append_summary to add the summary of the last {MAX_HISTORY_LENGTH} actions.")

        else:
            parts.append("What do you do next?")
        
        return "\n".join(parts)
    
    def _parse_response(self, response: str, valid_tools: list[str]) -> tuple[str, str, str, dict]:
        """Parse the LLM response to extract thought, summary, tool, and arguments."""
        thought = "No reasoning provided"
        result_summary = ""
        tool_name = "play_action"
        tool_args = {"action": "look"}
        
        lines = response.strip().split("\n")
        
        for line in lines:
            line_clean = line.strip()
            line_upper = line_clean.upper()
            
            if line_upper.startswith("THOUGHT:"):
                thought = line_clean.split(":", 1)[1].strip()

            elif line_upper.startswith("RESULT_SUMMARY:"):
                result_summary = line_clean.split(":", 1)[1].strip()

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
        
        return thought, result_summary,tool_name, tool_args

    def _extract_location(self, observation: str, fallback: str = "Unknown") -> str:
        """Extract location name from an observation, with fallback for non-location messages."""
        if not observation:
            return fallback

        first_line = observation.split("\n", 1)[0].strip()
        if not first_line:
            return fallback

        if self._is_likely_location_name(first_line):
            return first_line

        return fallback

    def _is_likely_location_name(self, text: str) -> bool:
        """Heuristic filter for room/location titles vs status/error messages."""
        if not text:
            return False

        lower = text.lower().strip()
        blocked_starts = [
            "you ",
            "there ",
            "the ",
            "opening ",
            "with ",
            "taken",
            "dropped",
        ]
        if any(lower.startswith(prefix) for prefix in blocked_starts):
            return False

        blocked_contains = [
            "can't",
            "cannot",
            "locked",
            "reveals",
            "nothing special",
        ]
        if any(token in lower for token in blocked_contains):
            return False

        if any(mark in text for mark in [".", "!", "?"]):
            return False

        if len(text) > 80:
            return False

        return True

    def _is_movement_action(self, action: str) -> bool:
        """Return True when the command is a movement/navigation action."""
        if not action:
            return False

        action = action.strip().lower()
        movement_commands = {
            "north", "south", "east", "west", "up", "down",
            "n", "s", "e", "w", "u", "d",
            "enter", "exit", "in", "out",
        }

        if action in movement_commands:
            return True

        if action.startswith("go "):
            direction = action[3:].strip()
            return direction in movement_commands

        return False
    
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

        if tool_name == "append_notepad":
            note = tool_args.get("note", "")
            if not isinstance(note, str):
                note = str(note)
            tool_args = {"note": note.strip()}

        if tool_name == "replace_in_notepad":
            old_string = tool_args.get("old_string", "")
            new_string = tool_args.get("new_string", "")
            if not isinstance(old_string, str):
                old_string = str(old_string)
            if not isinstance(new_string, str):
                new_string = str(new_string)
            tool_args = {
                "old_string": old_string,
                "new_string": new_string,
            }
        
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
