import os
from typing import Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class GeminiAssistant:
   # INITIALIZE 
   #Requires api key and model to be set in the .env file or passed as arguments
    MODELS_PREFERRED = [   
        "models/gemini-3.8-flash",
        "models/gemini-3.5-flash",
        "models/gemini-3.5-flash-lite",
        "models/gemini-flash-latest",
     ]

    
    #Initialize the GeminiAssistant with optional API key, model, and system prompt.
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        resolved_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not resolved_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing. Please add it to your .env file or pass it as an argument."
            )

        self.client = genai.Client(api_key=resolved_key)
        self.system_prompt = system_prompt or (
            "Please answer the user's question accurately and concisely. "
            "If you don't know the answer, please say 'I don't know'."
        )

        self.model = model or self._select_model()
        if not self.model:
            raise ValueError("No valid model found.")

    # Select the best available model from the API list.
    def _select_model(self) -> str:
        """Selecteert het beste beschikbare model uit de API lijst."""
        try:
            models_list = self.client.models.list()
            available_models = {m.name for m in models_list}

            for preferred_model in self.MODELS_PREFERRED:
                if preferred_model in available_models:
                    return preferred_model

            flash_models = [m for m in available_models if "flash" in m.lower()]
            if flash_models:
                return flash_models[0]

            if available_models:
                return next(iter(available_models))

        except Exception as e:
            print(f"Waarschuwing bij ophalen modellenlijst ({e}). Gebruikt standaardmodel.")

        return "models/gemini-3.8-flash"
    
    # Generate a response based on the provided prompt text.
    def generate_response(self, query:str, prompt_text: str) -> str:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        
        if not prompt_text or not prompt_text.strip():
            raise ValueError("Prompt cannot be empty.")

        #Create full_prompt for generating response
        full_prompt = f'''
        You are a highly intelligent AI assistant that provides accurate and concise answers based on the provided context. If the answer is not in the context, respond with "I don't know". 
        Context: {prompt_text}
        Question: {query}
        Answer:
        '''  

        # Temperatuur van 1.0 (of weglaten) is door Google aanbevolen bij Grounding
        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.7,
        )

        # Probeer primair model; bij 429 quota-fout valt hij automatisch terug op alternatieven
        fallback_models = [self.model] + [m for m in getattr(self, 'MODELS_PREFERRED', []) if m != self.model]

        response = None
        last_exception = None

        for current_model in fallback_models:
            try:
                response = self.client.models.generate_content(
                    model=current_model,
                    contents=full_prompt.strip(),
                    config=config,
                )
                self.model = current_model
                break
            except Exception as e:
                print(f"Fout bij model {current_model}: {e}. Probeer volgend model.")
             
            if not response:
              
              raise RuntimeError(f"Kan geen antwoord genereren: {last_exception}")

            answer = response.text or ""
            web_sources = []

            # Extraheer grounding bronnen als een schone lijst
            grounding_meta = getattr(response.candidates[0], "grounding_metadata", None) if response.candidates else None
            
            if grounding_meta and getattr(grounding_meta, "grounding_chunks", None):
                sources_set = set()
                for chunk in grounding_meta.grounding_chunks:
                    web = getattr(chunk, "web", None)
                    if web:
                        uri = getattr(web, "uri", None)
                        title = getattr(web, "title", None) or uri
                        if uri:
                            sources_set.add(f"[{title}]({uri})")
                web_sources = list(sources_set)

            return answer, web_sources