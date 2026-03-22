import re
import ast
import math
import operator
from .base import BaseReasoningAgent
from typing import AsyncGenerator


def _safe_math_eval(expr: str) -> float:
    """Evaluate a math expression using AST parsing. No eval()."""
    _ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    _funcs = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "log": math.log, "sin": math.sin,
        "cos": math.cos, "tan": math.tan, "pi": math.pi, "e": math.e,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant: {node.value!r}")
        if isinstance(node, ast.BinOp):
            op = _ops.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _ops.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
            return op(_eval(node.operand))
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _funcs:
                func = _funcs[node.func.id]
                args = [_eval(a) for a in node.args]
                return func(*args) if callable(func) else func
            raise ValueError(f"Unknown function: {ast.dump(node.func)}")
        if isinstance(node, ast.Name):
            if node.id in _funcs:
                val = _funcs[node.id]
                if not callable(val):
                    return val
            raise ValueError(f"Unknown name: {node.id}")
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    tree = ast.parse(expr.strip(), mode="eval")
    return _eval(tree)


class ReActAgent(BaseReasoningAgent):
    def __init__(self, client, model):
        super().__init__(client, model)
        self.name = "ReActAgent"

    async def perform_tool_call(self, tool_name, tool_input):
        # Tools implementation (basic for now, can be expanded to use real integrations)
        if tool_name == "calculate":
            try:
                return str(_safe_math_eval(tool_input))
            except Exception as e:
                return f"Error calculating: {str(e)}"
        elif tool_name == "search":
            try:
                from ai_congress.integrations.web_search import WebSearchEngine
                engine = WebSearchEngine()
                results = await engine.search(tool_input, max_results=3)
                if results:
                    return "\n".join(f"- {r.get('title', '')}: {r.get('snippet', '')}" for r in results)
                return f"No results found for: {tool_input}"
            except Exception as e:
                return f"Search error: {e}. Simulated results for: {tool_input}"
        else:
            return "Unknown tool"

    async def run(self, query: str) -> str:
        full_res = ""
        async for chunk in self.stream(query):
            full_res += chunk

        # Extract final answer if present
        match = re.search(r"Final Answer: (.*)", full_res, re.DOTALL)
        if match:
             return match.group(1).strip()
        return full_res

    async def stream(self, query: str) -> AsyncGenerator[str, None]:
        system_prompt = """You are a Reasoning and Acting agent.
Your goal is to answer the user question using tools if necessary.
Available tools:
- calculate: evaluates a mathematical expression (e.g., "3+3")
- search: simulates a web search (e.g., "population of france")

Format your response as:
Thought: <reasoning>
Action: <tool_name>[<input>]
Observation: <result>
... (repeat)
Thought: <reasoning>
Final Answer: <the final result>

Stop when you have the Final Answer.
"""
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': query}]
        max_steps = 5

        for i in range(max_steps):
            yield f"\n--- Step {i+1} ---\nAgent: "

            # In updated ollama client, we might need to handle context management manually or just keep appending messages
            # For simplicity, we just append to the prompt string if the model supports it,
            # OR we maintain the messages list.

            # Using messages list approach
            # Append prev observation if any?
            # Actually, let's keep it simple: we run the model, get text.

            current_response = ""
            async for chunk in await self.client.chat(self.model, messages, stream=True):
                 content = chunk['message']['content']
                 yield content
                 current_response += content

            # Add this turn to messages
            messages.append({'role': 'assistant', 'content': current_response})
            response_chunk = current_response

            if "Final Answer:" in response_chunk:
                return

            match = re.search(r"Action:\s*(\w+)\[(.*?)\]", response_chunk)
            if match:
                tool_name = match.group(1)
                tool_input = match.group(2)

                observation = await self.perform_tool_call(tool_name, tool_input)
                yield f"\nObservation: {observation}\n"

                messages.append({'role': 'user', 'content': f"Observation: {observation}"})
            else:
                # If no action, maybe it just forgot to formatted it or is done?
                # If it didn't say final answer but stopped, we might need to prompt it to continue.
                # For this demo, we'll assume if it stops without action, it's done or we break.
                if "Action:" not in response_chunk and "Final Answer:" not in response_chunk:
                     # Maybe it's just thinking?
                     yield "\n(No action detected, stopping)\n"
                     return
