import os
import time
import json
from typing import Dict, Any, List, Optional
import requests
from huggingface_hub import InferenceClient

class LLMLayer:
    """
    Enhanced LLM Layer with better handling of missing context and review queries.
    """
    
    def __init__(self):
        hf_token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
        
        if hf_token:
            self.hf_client = InferenceClient(token=hf_token)
            print("✅ HuggingFace client initialized with token")
        else:
            self.hf_client = InferenceClient()
            print("⚠️ Warning: No HUGGINGFACE_TOKEN found. API calls may be rate-limited.")
        
        self.available_models = {
            'zephyr-7b': 'HuggingFaceH4/zephyr-7b-beta',
            'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.2',
            'llama-3.2-3b': 'meta-llama/Llama-3.2-3B-Instruct',
            'neural-chat': 'Intel/neural-chat-7b-v3-3',
            'gemma-2b': 'google/gemma-2-2b-it',
        }
    
    def create_prompt(self, context: Optional[Dict[str, Any]], query: str, 
                     persona: str = "helpful travel assistant") -> str:
        """Enhanced prompt creation with better fallback handling."""
        
        if context is None:
            context = {}
        
        # Check if context has any data
        has_data = self._has_usable_data(context)
        
        # Format context
        context_str = self._format_context(context) if has_data else None
        
        # Build prompt based on data availability
        if has_data:
            prompt = f"""You are a {persona}.

CONTEXT (Retrieved from Knowledge Graph):
{context_str}

USER QUERY: {query}

TASK: Answer the user's query using ONLY the information provided in the context above. 
- Be specific and accurate
- If the context doesn't contain enough information, say so clearly
- Include relevant details like ratings, locations, and specific attributes
- For review queries, include specific feedback from travellers
- Format your response in a clear, helpful manner

ANSWER:"""
        else:
            # No data available - acknowledge it
            prompt = f"""You are a {persona}.

USER QUERY: {query}

CONTEXT: I searched the knowledge graph but could not find specific data matching your query.

TASK: Politely inform the user that:
1. The database doesn't contain information for their specific query
2. Suggest they try:
   - Searching for a different city (available cities can be queried)
   - Being more specific about hotel names
   - Asking for general recommendations
3. Offer to help with alternative queries

ANSWER:"""
        
        return prompt
    
    def _has_usable_data(self, context: Dict[str, Any]) -> bool:
        """Check if context contains usable data."""
        if not context:
            return False
        
        # Check baseline data
        baseline = context.get('baseline', {})
        if baseline and isinstance(baseline, dict):
            data = baseline.get('data', [])
            if data and len(data) > 0:
                return True
        
        # Check embedding data
        embedding = context.get('embedding', {})
        if embedding and isinstance(embedding, dict):
            data = embedding.get('data', [])
            if data and len(data) > 0:
                return True
        
        return False
    
    def _format_context(self, context: Optional[Dict[str, Any]]) -> str:
        """Enhanced context formatting with review support."""
        formatted = []
        
        if not context:
            return "No relevant data found in the knowledge graph."
        
        # Add intents and entities
        intents = context.get('intents', [])
        entities = context.get('entities', {})
        
        if intents:
            formatted.append(f"Detected Intent: {', '.join(intents)}")
        if entities:
            # Filter out None values
            filtered_entities = {k: v for k, v in entities.items() if v is not None}
            if filtered_entities:
                formatted.append(f"Extracted Entities: {filtered_entities}")
        if intents or entities:
            formatted.append("")
        
        # Check if fallback was used
        baseline_data = context.get('baseline', {})
        if baseline_data.get('fallback_used'):
            formatted.append("⚠️ NOTE: No specific location was found, showing global results.")
            formatted.append("")
        
        # Format baseline data
        if baseline_data and isinstance(baseline_data, dict):
            data = baseline_data.get('data', [])
            if data:
                # Detect data type
                if self._is_review_data(data):
                    formatted.append("=== REVIEWS FROM KNOWLEDGE GRAPH ===")
                    for i, item in enumerate(data[:10], 1):
                        formatted.append(f"\n{i}. {self._format_review(item)}")
                else:
                    formatted.append("=== HOTEL DATA FROM KNOWLEDGE GRAPH ===")
                    for i, item in enumerate(data[:10], 1):
                        formatted.append(f"\n{i}. {self._format_hotel(item)}")
                formatted.append("")
        
        # Format embedding data
        embedding_data = context.get('embedding')
        if embedding_data and isinstance(embedding_data, dict):
            data = embedding_data.get('data', [])
            if data:
                formatted.append("=== SEMANTIC SEARCH RESULTS ===")
                for i, item in enumerate(data, 1):
                    similarity = item.get('similarity', 0)
                    formatted.append(f"\n{i}. (Similarity: {similarity:.3f}) {self._format_hotel(item)}")
                formatted.append("")
        
        if not formatted:
            return "No relevant data found in the knowledge graph."
        
        return "\n".join(formatted)
    
    def _is_review_data(self, data: List[Dict[str, Any]]) -> bool:
        """Check if data contains reviews."""
        if not data:
            return False
        first_item = data[0]
        return 'review_id' in first_item or 'text' in first_item
    
    def _format_review(self, item: Dict[str, Any]) -> str:
        """Format a review item."""
        parts = []
        
        if 'hotel_name' in item:
            parts.append(f"Hotel: {item['hotel_name']}")
        if 'city' in item:
            parts.append(f"City: {item['city']}")
        if 'overall_score' in item and item['overall_score']:
            parts.append(f"Rating: {item['overall_score']:.1f}/10")
        if 'traveller_type' in item:
            parts.append(f"Traveller: {item['traveller_type']}")
        if 'gender' in item and item['gender']:
            parts.append(f"Gender: {item['gender']}")
        if 'age' in item and item['age']:
            parts.append(f"Age: {item['age']}")
        if 'date' in item:
            parts.append(f"Date: {item['date']}")
        
        # Add review text
        if 'text' in item and item['text']:
            text = item['text']
            if len(text) > 300:
                text = text[:300] + "..."
            parts.append(f"\nReview: \"{text}\"")
        
        # Add detailed scores if available
        scores = []
        if 'cleanliness' in item and item['cleanliness']:
            scores.append(f"Cleanliness: {item['cleanliness']:.1f}")
        if 'comfort' in item and item['comfort']:
            scores.append(f"Comfort: {item['comfort']:.1f}")
        if 'facilities' in item and item['facilities']:
            scores.append(f"Facilities: {item['facilities']:.1f}")
        
        if scores:
            parts.append("\nScores: " + ", ".join(scores))
        
        return " | ".join(parts) if parts else "No data available"
    
    def _format_hotel(self, item: Dict[str, Any]) -> str:
        """Format a hotel item."""
        parts = []
        
        if 'name' in item:
            parts.append(f"Hotel: {item['name']}")
        if 'city' in item:
            parts.append(f"City: {item['city']}")
        if 'country' in item:
            parts.append(f"Country: {item['country']}")
        if 'avg_score' in item and item['avg_score']:
            parts.append(f"Rating: {item['avg_score']:.2f}/10")
        if 'star_rating' in item or 'stars' in item:
            stars = item.get('star_rating') or item.get('stars')
            if stars:
                parts.append(f"Stars: {stars}")
        
        # Facility scores
        if 'cleanliness' in item and item['cleanliness']:
            parts.append(f"Cleanliness: {item['cleanliness']:.1f}/10")
        if 'comfort' in item and item['comfort']:
            parts.append(f"Comfort: {item['comfort']:.1f}/10")
        
        return " | ".join(parts) if parts else "No data available"
    
    def _create_smart_fallback(self, prompt: str, model_name: str, elapsed_time: float) -> Dict[str, Any]:
        """Create intelligent fallback response."""
        user_query = "your query"
        if "USER QUERY:" in prompt:
            query_start = prompt.find("USER QUERY:") + len("USER QUERY:")
            query_end = prompt.find("TASK:", query_start)
            if query_end > query_start:
                user_query = prompt[query_start:query_end].strip()
        
        # Extract context data
        context_data = []
        if "=== HOTEL DATA" in prompt or "=== REVIEWS" in prompt:
            # Try to extract some data
            lines = prompt.split('\n')
            for line in lines:
                if line.strip() and any(char.isdigit() for char in line[:5]):
                    context_data.append(line.strip())
        
        # Generate response
        if context_data:
            response = f"Based on your query '{user_query}', here's what I found:\n\n"
            for i, item in enumerate(context_data[:5], 1):
                item_clean = item.split('. ', 1)[-1] if '. ' in item else item
                response += f"{i}. {item_clean}\n\n"
            
            if len(context_data) > 5:
                response += f"...and {len(context_data) - 5} more results.\n\n"
            
            response += "Note: API connection issue. Using fallback formatting."
        else:
            response = f"I couldn't find specific data for '{user_query}'. This might be because:\n\n"
            response += "1. The location/hotel is not in the database\n"
            response += "2. The query needs to be more specific\n"
            response += "3. No matching data exists\n\n"
            response += "Try asking about available cities or being more specific."
        
        return {
            'model': model_name,
            'response': response,
            'tokens_used': 0,
            'time_seconds': elapsed_time,
            'cost': 0.0,
            'success': False,
            'error': 'API methods exhausted, using fallback'
        }
    
    def query_huggingface(self, prompt: str, model_name: str = 'mistral-7b',
                         max_tokens: int = 500) -> Dict[str, Any]:
        """Query HuggingFace model with multiple fallback methods."""
        start_time = time.time()
        model_id = self.available_models[model_name]
        
        print(f"🔍 Calling HuggingFace API with model: {model_id}")
        
        # Try Method 1: chat_completion
        try:
            print("  → Trying chat_completion method...")
            response = self.hf_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model_id,
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            if hasattr(response, 'choices') and len(response.choices) > 0:
                response_text = response.choices[0].message.content
                elapsed_time = time.time() - start_time
                print(f"✅ chat_completion succeeded in {elapsed_time:.2f}s")
                
                return {
                    'model': model_name,
                    'response': response_text,
                    'tokens_used': len(prompt.split()) + len(response_text.split()),
                    'time_seconds': elapsed_time,
                    'cost': 0.0,
                    'success': True
                }
        except Exception as e1:
            print(f"  ✗ chat_completion failed: {e1}")
        
        # Try Method 2: text_generation
        try:
            print("  → Trying text_generation method...")
            response = self.hf_client.text_generation(
                prompt,
                model=model_id,
                max_new_tokens=max_tokens,
                temperature=0.7,
                return_full_text=False
            )
            
            if response:
                elapsed_time = time.time() - start_time
                print(f"✅ text_generation succeeded in {elapsed_time:.2f}s")
                
                return {
                    'model': model_name,
                    'response': response,
                    'tokens_used': len(prompt.split()) + len(str(response).split()),
                    'time_seconds': elapsed_time,
                    'cost': 0.0,
                    'success': True
                }
        except Exception as e2:
            print(f"  ✗ text_generation failed: {e2}")
        
        # Try Method 3: Direct API call
        try:
            print("  → Trying direct API call...")
            api_url = f"https://api-inference.huggingface.co/models/{model_id}"
            headers = {}
            
            token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
            if token:
                headers["Authorization"] = f"Bearer {token}"
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": 0.7,
                    "return_full_text": False
                }
            }
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                if isinstance(result, list) and len(result) > 0:
                    response_text = result[0].get('generated_text', str(result))
                elif isinstance(result, dict):
                    response_text = result.get('generated_text', str(result))
                else:
                    response_text = str(result)
                
                elapsed_time = time.time() - start_time
                print(f"✅ Direct API call succeeded in {elapsed_time:.2f}s")
                
                return {
                    'model': model_name,
                    'response': response_text,
                    'tokens_used': len(prompt.split()) + len(response_text.split()),
                    'time_seconds': elapsed_time,
                    'cost': 0.0,
                    'success': True
                }
        except Exception as e3:
            print(f"  ✗ Direct API call failed: {e3}")
        
        # All methods failed
        elapsed_time = time.time() - start_time
        print(f"❌ All API methods failed")
        return self._create_smart_fallback(prompt, model_name, elapsed_time)
    
    def query_model(self, prompt: str, model_name: str = 'mistral-7b',
                   max_tokens: int = 500) -> Dict[str, Any]:
        """Universal query method."""
        hf_models = ['mistral-7b', 'gemma-2b', 'llama-3.2-3b', 
                     'zephyr-7b', 'neural-chat']
        
        if model_name in hf_models:
            return self.query_huggingface(prompt, model_name, max_tokens)
        else:
            return {
                'model': model_name,
                'response': f"Error: Unknown model {model_name}",
                'success': False
            }
    def compare_models(self, context: Optional[Dict[str, Any]], query: str,
                      models: List[str] = None) -> Dict[str, Any]:
        """Compare multiple models on the same query."""
        if models is None:
            # Use free models by default
            models = ['mistral-7b', 'gemma-2b', 'phi-3']
        
        prompt = self.create_prompt(context, query)
        
        results = {}
        for model in models:
            print(f"Querying {model}...")
            result = self.query_model(prompt, model)
            results[model] = result
        
        # Create comparison summary
        comparison = {
            'query': query,
            'models_tested': models,
            'results': results,
            'summary': self._create_comparison_summary(results)
        }
        
        return comparison
    
    def _create_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create quantitative comparison summary."""
        summary = {
            'fastest_model': None,
            'cheapest_model': None,
            'longest_response': None,
            'response_times': {},
            'costs': {},
            'response_lengths': {}
        }
        
        for model, result in results.items():
            if not result.get('success'):
                continue
            
            time_taken = result.get('time_seconds', float('inf'))
            cost = result.get('cost', 0.0)
            response_length = len(result.get('response', ''))
            
            summary['response_times'][model] = time_taken
            summary['costs'][model] = cost
            summary['response_lengths'][model] = response_length
        
        # Find fastest, cheapest, longest
        if summary['response_times']:
            summary['fastest_model'] = min(summary['response_times'], 
                                          key=summary['response_times'].get)
        if summary['costs']:
            summary['cheapest_model'] = min(summary['costs'], 
                                           key=summary['costs'].get)
        if summary['response_lengths']:
            summary['longest_response'] = max(summary['response_lengths'], 
                                             key=summary['response_lengths'].get)
        
        return summary


# Compatibility wrapper
class LLMManager(LLMLayer):
    """Compatibility wrapper - redirects to LLMLayer"""
    
    def __init__(self):
        super().__init__()
        print("ℹ️ Note: LLMManager is deprecated. Use LLMLayer instead.")
