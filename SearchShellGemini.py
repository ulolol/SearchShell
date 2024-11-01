import requests
from bs4 import BeautifulSoup
from googlesearch import search
import cmd
import shlex
import time
import json
import toml
from typing import List, Dict

class SearchShell(cmd.Cmd):
    intro = 'Welcome to the Search Shell. Type search, followed by your Query. Type help or ? to list commands.\n'
    prompt = 'search> '
    
    def __init__(self):
        super().__init__()
        self.wrapper = GeminiWebWrapper()
        self.show_context = False
        self.num_results = 3

    def do_search(self, arg):
        """Search and get response. Usage: search <query> [--show-context] [--results=<num>]"""
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
                        elif args[i] == '--results' and i + 1 < len(args):
                            self.num_results = int(args[i + 1])
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

            print(f"\nSearching web using Gemini 1.5 Flash...")
            context = self.wrapper.generate_context(query, self.num_results)

            if self.show_context:
                print("\nContext gathered from web:")
                print(context)
                print("\nGenerating response...")

            response = self.wrapper.query_gemini(query, context)
            print("\nResponse:")
            print(response)
            print()  # Extra newline for readability

        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            self.show_context = False
            self.num_results = 3

    def do_config(self, arg):
        """Show or set default configuration. Usage: config [show|set <param> <value>]"""
        args = shlex.split(arg)
        if not args or args[0] == 'show':
            print(f"Current configuration:")
            print("Current model: Gemini 1.5 Flash")
            print(f"Number of results: {self.num_results}")
            print(f"Show context: {self.show_context}")
        elif args[0] == 'set' and len(args) >= 3:
            param = args[1]
            value = args[2]
            if param == 'results':
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


class GeminiWebWrapper:
    def __init__(self):
        self.api_key = self.load_api_key()
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-8b:generateContent"

    def load_api_key(self) -> str:
        """Load the Gemini API key from a TOML file."""
        try:
            config = toml.load("SearchShellGPT.toml")
            return config['gemini']['api_key']
        except Exception as e:
            raise RuntimeError("Failed to load API key from SearchShellGPT.toml") from e

    def search_web(self, query: str, num_results: int = 3) -> List[Dict]:
        """Search the web using Google and return results."""
        try:
            results = []
            for url in search(query, num_results=num_results):
                results.append({
                    'href': url,
                    'title': self._get_page_title(url),
                    'body': ''
                })
            return results
        except Exception as e:
            print(f"Error searching web: {e}")
            return []

    def _get_page_title(self, url: str) -> str:
        """Extract the title from a webpage."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else url
            return title.strip()
        except Exception:
            return url

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
            url = result['href']
            title = result['title']

            print(f"\nFetching content from: {url}")
            content = self.extract_content(url)

            if content:
                context_entry = (
                    f"Source: {title}\n"
                    f"URL: {url}\n\n"
                    f"Content:\n{content}\n"
                    f"{'='*50}\n"
                )
                context.append(context_entry)
            else:
                context_entry = (
                    f"Source: {title}\n"
                    f"URL: {url}\n"
                    f"{'='*50}\n"
                )
                context.append(context_entry)

            time.sleep(1)

        return "\n".join(context)

    def query_gemini(self, prompt: str, context: str) -> str:
        """Query Gemini with the given prompt and context using direct API call."""
        try:
            if not context.strip():
                return "No context was found from web searches. The model will provide a general response without current information."

            full_prompt = (
                f"Context from web searches:\n\n{context}\n\n"
                f"Question: {prompt}\n\n"
                f"Please provide a comprehensive answer based on the context above. "
                f"Please ensure the answer is detailed with points wherever necessary. "
                f"Please ensure that the answer is properly formatted for reading. "
                f"If the context doesn't contain relevant information, please state that clearly."
            )

            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }

            data = {
                "contents": [{
                    "parts": [{
                        "text": full_prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 2048,
                }
            }

            response = requests.post(
                f"{self.api_url}?key={self.api_key}",
                headers=headers,
                json=data,
                timeout=30
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            # Extract the generated text from the response
            try:
                return response_data['candidates'][0]['content']['parts'][0]['text']
            except (KeyError, IndexError) as e:
                return f"Error parsing Gemini response: {str(e)}"

        except requests.exceptions.RequestException as e:
            return f"Error querying Gemini API: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

def main():
    SearchShell().cmdloop()

if __name__ == "__main__":
    main()