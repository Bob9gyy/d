"""
plugin_system.py — Extensible plugin architecture for JARVIS
Allows adding new tools without modifying core code.
"""
from __future__ import annotations

import importlib.util
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class BaseTool(ABC):
    """Base class for all plugin tools."""
    
    name: str
    description: str
    risk_level: str = "MEDIUM"
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given arguments. Must return a string result."""
        pass
    
    def get_schema(self) -> dict:
        """Return schema of tool arguments for LLM planning."""
        return {
            "name": self.name,
            "description": self.description,
            "risk_level": self.risk_level,
            "parameters": {}
        }


class PluginSystem:
    """Manages plugin discovery, loading, and execution."""
    
    def __init__(self, plugin_dir: str = "./plugins"):
        self.plugin_dir = Path(plugin_dir)
        self.plugin_dir.mkdir(exist_ok=True)
        
        self.tools: dict[str, BaseTool] = {}
        self.schemas: dict[str, dict] = {}
    
    def discover_plugins(self) -> list[str]:
        """Discover all available plugins."""
        plugins = []
        for plugin_file in self.plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
            plugins.append(plugin_file.stem)
        return plugins
    
    def load_plugin(self, plugin_name: str) -> bool:
        """Load a single plugin."""
        try:
            plugin_file = self.plugin_dir / f"{plugin_name}.py"
            
            if not plugin_file.exists():
                print(f"Plugin file not found: {plugin_file}")
                return False
            
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseTool) and 
                    obj is not BaseTool):
                    
                    tool = obj()
                    self.tools[tool.name] = tool
                    self.schemas[tool.name] = tool.get_schema()
                    print(f"Loaded plugin tool: {tool.name}")
            
            return True
        
        except Exception as e:
            print(f"Failed to load plugin {plugin_name}: {e}")
            return False
    
    def load_all_plugins(self) -> int:
        """Load all available plugins. Returns count of successfully loaded."""
        plugins = self.discover_plugins()
        loaded = 0
        
        for plugin_name in plugins:
            if self.load_plugin(plugin_name):
                loaded += 1
        
        return loaded
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a registered tool."""
        if tool_name not in self.tools:
            return f"Unknown tool: {tool_name}"
        
        try:
            tool = self.tools[tool_name]
            return tool.execute(**kwargs)
        except Exception as e:
            return f"Tool error: {e}"
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a registered tool."""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> list[str]:
        """List all registered tools."""
        return list(self.tools.keys())
    
    def get_all_schemas(self) -> dict[str, dict]:
        """Get schemas for all registered tools."""
        return self.schemas.copy()
