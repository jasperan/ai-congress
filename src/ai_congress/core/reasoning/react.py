import re
import math
from .base import BaseReasoningAgent
from typing import AsyncGenerator

class ReActAgent(BaseReasoningAgent):
    def __init__(self, client, model):
        super().__init__(client, model)
        self.name = "ReActAgent"
    
    def perform_tool_call(self, tool_name, tool_input):
        # Tools implementation (basic for now, can be expanded to use real integrations)
        if tool_name == "calculate":
            try:
                # Safe eval
                allowed_names = {"math": math, "abs": abs, "round": round, "min": min, "max": max}
                return str(eval(tool_input, {"__builtins__": None}, allowed_names))
            except Exception as e:
                return f"Error calculating: {str(e)}"
        elif tool_name == "search":
             # This acts as a placeholder if real web search isn't injected. 
             # Ideally this would call the web_search integrations.
             # For now we simulate basic knowledge or return a hint to use built-in knowledge
            return f"Simulated search results for: {tool_input}. (For real search, enable web_search)"
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
        full_conversation = f"{system_prompt}\nQuestion: {query}\n"
        
        for i in range(max_steps):
            yield f"\n--- Step {i+1} ---\nAgent: "
            
            response_chunk = ""
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
                
                observation = self.perform_tool_call(tool_name, tool_input)
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
