import argparse
import requests
import json
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import ollama
import sys
from typing import List, Dict
import time

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
            response.raise_for_status()  # Raise an exception for bad status codes
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'iframe', 'noscript']):
                element.decompose()
            
            # Focus on main content areas
            main_content = soup.find('main') or soup.find('article') or soup.find('div', {'class': ['content', 'main', 'article']})
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)
            
            # Basic text cleaning
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            cleaned_text = '\n'.join(lines)
            
            # Truncate if too long
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
                    # Fallback to just using the snippet
                    context_entry = (
                        f"Source: {title}\n"
                        f"URL: {url}\n"
                        f"Summary: {snippet}\n"
                        f"{'='*50}\n"
                    )
                    context.append(context_entry)
                
                # Add a small delay between requests
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
    parser = argparse.ArgumentParser(description="Ollama wrapper with web search capabilities")
    parser.add_argument("query", help="The query to process")
    parser.add_argument("--model", default="granite3-moe:3b-instruct-q8_0", help="Ollama model to use")
    parser.add_argument("--results", type=int, default=3, help="Number of web results to fetch")
    parser.add_argument("--show-context", action="store_true", help="Show the context being sent to Ollama")
    
    args = parser.parse_args()
    
    wrapper = OllamaWebWrapper(model_name=args.model)
    
    print("Searching web and generating context...")
    context = wrapper.generate_context(args.query, args.results)
    
    if args.show_context:
        print("\nContext gathered from web:")
        print(context)
        print("\nGenerating response...")
    
    response = wrapper.query_ollama(args.query, context)
    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    main()
