import argparse
import requests
import json
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import ollama
import sys
from typing import List, Dict
import time
import cmd
import shlex

class SearchShell(cmd.Cmd):
    intro = 'Welcome to the Search Shell. Type search, followed by your Query. Type help or ? to list commands.\n'
    prompt = 'search> '

    def __init__(self):
        super().__init__()
        self.wrapper = OllamaWebWrapper()
        self.show_context = False
        self.num_results = 3
        self.model = "granite3-moe:3b-instruct-q8_0"

    def do_search(self, arg):
        """Search and get response. Usage: search <query> [--show-context] [--results=<num>] [--model=<model>]"""
        try:
            if not arg.strip():
                print("Please provide a search query")
                return

            args = []
            query_parts = []

            try:
                args = shlex.split(arg)
            except ValueError as e:
                query_parts = [arg]
            else:
                i = 0
                while i < len(args):
                    if args[i].startswith('--'):
                        if args[i] == '--show-context':
                            self.show_context = True
                        elif args[i].startswith('--results='):
                            self.num_results = int(args[i].split('=')[1])
                        elif args[i].startswith('--model='):
                            self.model = args[i].split('=')[1]
                        elif args[i] == '--results' and i + 1 < len(args):
                            self.num_results = int(args[i + 1])
                            i += 1
                        elif args[i] == '--model' and i + 1 < len(args):
                            self.model = args[i + 1]
                            i += 1
                        else:
                            query_parts.append(args[i])
                    else:
                        query_parts.append(args[i])
                    i += 1

            query = ' '.join(query_parts)
            if not query:
                print("Please provide a search query")
                return

            self.wrapper.model_name = self.model
            print(f"\nSearching web using model '{self.model}'...")
            context = self.wrapper.generate_context(query, self.num_results)

            if self.show_context:
                print("\nContext gathered from web:")
                print(context)
                print("\nGenerating response...")

            response = self.wrapper.query_ollama(query, context)
            print("\nResponse:")
            print(response)
            print()  # Extra newline for readability

        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            self.show_context = False
            self.num_results = 3
            self.model = "granite3-moe:3b-instruct-q8_0"

    def do_config(self, arg):
        """Show or set default configuration. Usage: config [show|set <param> <value>]"""
        args = shlex.split(arg)
        if not args or args[0] == 'show':
            print(f"Current configuration:")
            print(f"Model: {self.model}")
            print(f"Number of results: {self.num_results}")
            print(f"Show context: {self.show_context}")
        elif args[0] == 'set' and len(args) >= 3:
            param = args[1]
            value = args[2]
            if param == 'model':
                self.model = value
            elif param == 'results':
                self.num_results = int(value)
            elif param == 'show_context':
                self.show_context = value.lower() == 'true'
            print(f"Updated {param} to {value}")
        else:
            print("Invalid config command. Use 'config show' or 'config set <param> <value>'")

    def do_exit(self, arg):
        """Exit the search shell"""
        print("Goodbye!")
        return True

    def do_quit(self, arg):
        """Exit the search shell"""
        return self.do_exit(arg)

    def do_EOF(self, arg):
        """Exit on Ctrl-D"""
        print()  # Empty line before exiting
        return True

    # Shortcuts for common commands
    do_q = do_quit
    do_s = do_search

    def emptyline(self):
        """Do nothing on empty line"""
        pass

class OllamaWebWrapper:
    def __init__(self, model_name: str = "granite3-moe:3b-instruct-q8_0"):
        self.model_name = model_name
        self.ddgs = DDGS()
        
    def search_web(self, query: str, num_results: int = 3) -> List[Dict]:
        """Search the web using DuckDuckGo and return results."""
        try:
            results = list(self.ddgs.text(query, max_results=num_results))
            return results
        except Exception as e:
            print(f"Error searching web: {e}")
            return []

    def extract_content(self, url: str) -> str:
        """Extract main content from a webpage."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'iframe', 'noscript']):
                element.decompose()
            
            main_content = soup.find('main') or soup.find('article') or soup.find('div', {'class': ['content', 'main', 'article']})
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)
            
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            cleaned_text = '\n'.join(lines)
            
            return cleaned_text[:2000]
        except Exception as e:
            print(f"Error extracting content from {url}: {e}")
            return ""

    def generate_context(self, query: str, num_results: int = 3) -> str:
        """Generate context from web search results."""
        results = self.search_web(query, num_results)
        context = []
        
        for result in results:
            url = result.get('href', '')
            title = result.get('title', '')
            snippet = result.get('body', '')
            
            if url:
                print(f"\nFetching content from: {url}")
                content = self.extract_content(url)
                if content:
                    context_entry = (
                        f"Source: {title}\n"
                        f"URL: {url}\n"
                        f"Summary: {snippet}\n\n"
                        f"Content:\n{content}\n"
                        f"{'='*50}\n"
                    )
                    context.append(context_entry)
                else:
                    context_entry = (
                        f"Source: {title}\n"
                        f"URL: {url}\n"
                        f"Summary: {snippet}\n"
                        f"{'='*50}\n"
                    )
                    context.append(context_entry)
                
                time.sleep(1)
        
        return "\n".join(context)

    def query_ollama(self, prompt: str, context: str) -> str:
        """Query Ollama with the given prompt and context."""
        try:
            if not context.strip():
                return "No context was found from web searches. The model will provide a general response without current information."
            
            full_prompt = (
                f"Context from web searches:\n\n{context}\n\n"
                f"Question: {prompt}\n\n"
                f"Please provide a comprehensive answer based on the context above. "
                f"If the context doesn't contain relevant information, please state that clearly."
            )
            
            response = ollama.chat(model=self.model_name, messages=[
                {
                    'role': 'user',
                    'content': full_prompt
                }
            ])
            return response['message']['content']
        except Exception as e:
            return f"Error querying Ollama: {e}"

def main():
    # Start the interactive shell
    SearchShell().cmdloop()

if __name__ == "__main__":
    main()
