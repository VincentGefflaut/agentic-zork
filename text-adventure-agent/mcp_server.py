"""
Student MCP Server for Text Adventure Games

This is your MCP server submission. Implement the tools that your agent
will use to play text adventure games.

Required tool:
    play_action(action: str) -> str
        Execute a game command and return the result.

Recommended tools:
    memory() -> str
        Return current game state, score, and recent history.
    
    inventory() -> str  
        Return the player's current inventory.
    
    get_map() -> str
        Return a map of explored locations.

Test your server with:
    fastmcp dev submission_template/mcp_server.py

Then open the MCP Inspector in your browser to test the tools interactively.
"""

import sys
import os

# Add parent directory to path to import games module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from games.zork_env import TextAdventureEnv


# Feature flags
NOTEPAD = True


# =============================================================================
# Create the MCP Server
# =============================================================================

mcp = FastMCP("Student Text Adventure Server")


# =============================================================================
# Game State Management
# =============================================================================

class GameManager:
    """
    Manages the text adventure game state.
    
    TODO: Extend this class to track:
    - Action history (for memory tool)
    - Explored locations (for mapping)
    - Current score and moves
    """
    
    def __init__(self):
        self.env: TextAdventureEnv = None
        self.state = None
        self.game_name: str = ""
        # TODO: Add more state tracking
        self.history: list[tuple[str, str]] = []
        self.explored_locations: dict[str, set[str]] = {}
        self.current_location: str = ""
        self.notepad: str = ""
    
    def initialize(self, game: str = "zork1"):
        """Initialize or reset the game."""
        self.game_name = game
        self.env = TextAdventureEnv(game)
        self.state = self.env.reset()
        # TODO: Reset your state tracking here
        self.history = []
        self.explored_locations = {}
        self.current_location = self._extract_location(
            self.state.observation,
            fallback="Starting point"
        )
        self.notepad = ""
        return self.state.observation
    
    def _extract_location(self, observation: str, fallback: str = "Unknown") -> str:
        """Extract likely location name from observation text."""
        if not observation:
            return fallback

        lines = [line.strip() for line in observation.split("\n") if line.strip()]
        for line in lines:
            if self._is_likely_location_name(line):
                return line

        return fallback

    def _is_likely_location_name(self, text: str) -> bool:
        """Heuristic check to distinguish room names from status messages."""
        if not text or len(text) > 80:
            return False

        if text.startswith("["):
            return False

        if any(mark in text for mark in [".", "!", "?", ",", ";", ":", '"']):
            return False

        if not text[0].isalpha() or not text[0].isupper():
            return False

        return True

    def _is_movement_action(self, action: str) -> bool:
        """Return True when the action is a movement/navigation command."""
        action = (action or "").strip().lower()
        if not action:
            return False

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
    
    def step(self, action: str) -> str:
        """Execute an action and return the result."""
        if self.env is None:
            self.initialize()
        
        self.state = self.env.step(action)
        
        # TODO: Update your state tracking here
        self.history.append((action, self.state.observation))

        # Update location tracking, etc.
        old_location = self.current_location or "Unknown"
        new_location = self._extract_location(self.state.observation, fallback=old_location)
        normalized_action = (action or "").strip().lower()

        if self._is_movement_action(normalized_action):
            if old_location not in self.explored_locations:
                self.explored_locations[old_location] = set()
            if new_location != old_location:
                self.explored_locations[old_location].add(f"{normalized_action} -> {new_location}")
                self.current_location = new_location
        elif not self.current_location:
            self.current_location = new_location
        
        return self.state.observation
    
    def get_score(self) -> int:
        """Get current score."""
        return self.state.score if self.state else 0
    
    def get_moves(self) -> int:
        """Get number of moves taken."""
        return self.state.moves if self.state else 0
    
    def get_inventory(self) -> str:
        """Get current inventory."""
        items = self.state.inventory if hasattr(self.state, 'inventory') and self.state.inventory else []
        
        if not items:
            return "Inventory: You are empty-handed."
        
        item_names = []
        for item in items:
            item_str = str(item)
            item_lower = item_str.lower()
            if "parent" in item_lower:
                idx = item_lower.index("parent")
                name = item_str[:idx].strip()
                if ":" in name:
                    name = name.split(":", 1)[1].strip()
                item_names.append(name)
            elif ":" in item_str:
                name = item_str.split(":")[1].strip()
                item_names.append(name)
            else:
                item_names.append(item_str)
        
        return f"Inventory: {', '.join(item_names)}"

    def get_map(self) -> str:
        """Get a map of explored locations."""
        if not self.explored_locations:
            return "Map: No locations explored yet. Try moving around!"
        
        lines = ["Explored Locations and Exits:"]
        for loc, exits in sorted(self.explored_locations.items(), reverse=True):
            lines.append(f"\n* {loc}")
            for exit_info in sorted(exits):
                lines.append(f"    -> {exit_info}")
        
        lines.append(f"\n[Current] {self.current_location}")
        return "\n".join(lines)

    def get_memory(self) -> str:
        """Get a summary of current game state."""
        recent = self.history[-5:] if self.history else []
        recent_str = "\n".join([f"  > {a} -> {r[:60]}..." for a, r in recent]) if recent else "  (none yet)"
        
        return f"""Current State:
- Location: {self.current_location}
- Score: {self.state.score} points
- Moves: {self.state.moves}
- Game: {self.game_name}

Recent Actions:
{recent_str}

Current Observation:
{self.state.observation}"""

    def append_notepad(self, note: str) -> str:
        """Append a note to the notepad and return full content."""
        note = (note or "").strip()
        if not note:
            return self.notepad
        if self.notepad:
            self.notepad += f"\n{note}"
        else:
            self.notepad = note
        return self.notepad

    def replace_in_notepad(self, old_string: str, new_string: str) -> str:
        """Replace content in the notepad and return full content."""
        old_string = old_string or ""
        new_string = new_string or ""
        if not old_string:
            return self.notepad
        self.notepad = self.notepad.replace(old_string, new_string)
        return self.notepad

    def get_notepad(self) -> str:
        """Return full notepad content."""
        return self.notepad

# Global game manager
_game = GameManager()


def get_game() -> GameManager:
    """Get or initialize the game manager."""
    global _game
    if _game.env is None:
        # Get game from environment variable (set by evaluator)
        game = os.environ.get("GAME", "zork1")
        _game.initialize(game)
    return _game


# =============================================================================
# MCP Tools - IMPLEMENT THESE
# =============================================================================

@mcp.tool()
def play_action(action: str) -> str:
    """
    Execute a game command and return the result.
    
    This is the main tool for interacting with the game.
    
    Args:
        action: The command to execute (e.g., "north", "take lamp", "open mailbox")
        
    Returns:
        The game's response to the action
        
    Valid commands include:
        - Movement: north, south, east, west, up, down, enter, exit
        - Objects: take <item>, drop <item>, open <thing>, examine <thing>
        - Other: look, inventory, read <thing>, turn on lamp
    """
    game = get_game()
    
    # TODO: You might want to add action validation here
    # TODO: You might want to include score changes in the response
    
    previous_score = game.get_score()
    result = game.step(action)
    current_score = game.get_score()
    score_diff = current_score - previous_score
    if score_diff > 0:
        result += f"\n[You gained {score_diff} points! Total score: {current_score}]"
    elif score_diff < 0:
        result += f"\n[You lost {-score_diff} points. Total score: {current_score}]"
    
    return result

# TODO: Implement additional tools to help your agent

@mcp.tool()
def memory() -> str:
    """
    Get a summary of the current game state.
    
    Returns location, score, moves, recent actions, and current observation.
    """
    return get_game().get_memory()


@mcp.tool()
def inventory() -> str:
    """
    Check what the player is carrying.
    
    Returns:
        List of items in the player's inventory
    """
    game = get_game()
    result = game.step("inventory")
    return result


@mcp.tool()
def get_map() -> str:
    """
    Get a map showing explored locations and connections.
    
    Useful for navigation and avoiding getting lost.
    """
    return get_game().get_map()


@mcp.tool()
def get_valid_actions() -> str:
    """
    Get a list of likely valid actions from the current location.
      
    Returns:
        List of actions that might work here
    """
    # This is a hint: Jericho provides get_valid_actions()
    game = get_game()
    if game.env and game.env.env:
        valid = game.env.env.get_valid_actions()
        return ", ".join(valid) if valid else ""
    return ""

@mcp.tool()
def append_summary(summary: str) -> str:
    """
    Append a summary of the current game state to the history.
    """
    return summary


if NOTEPAD:
    @mcp.tool()
    def append_notepad(note: str) -> str:
        """
        Append text to the notepad.

        Args:
            note: Text to append as a new line

        Returns:
            Full notepad content after update
        """
        return get_game().append_notepad(note)


    @mcp.tool()
    def replace_in_notepad(old_string: str, new_string: str) -> str:
        """
        Replace text in the notepad.

        Args:
            old_string: Existing text to replace
            new_string: Replacement text

        Returns:
            Full notepad content after replacement
        """
        return get_game().replace_in_notepad(old_string, new_string)


# =============================================================================
# Run the server
# =============================================================================

if __name__ == "__main__":
    # This runs the server with stdio transport (for MCP clients)
    mcp.run()
