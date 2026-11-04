import os
from typing import Optional, Tuple, List
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class GeminiAssistant:
    """
    Wrapper om via de officiële `google-genai` SDK antwoorden te genereren 
    met ondersteuning voor model-fallbacks en Google Search Grounding.
    """

    MODELS_PREFERRED = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
        ]
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
            "You are a highly intelligent AI assistant that provides accurate and concise answers "
            "based on the provided context. If the answer is not in the context, respond with 'I don't know'."
        )

        # Verwijder eventuele 'models/' prefix voor de nieuwe SDK
        raw_model = model or self._select_model()
        self.model = raw_model.replace("models/", "") if raw_model else "gemini-2.5-flash"

    def _select_model(self) -> str:
        """Selecteert het beste beschikbare model uit de API-lijst."""
        try:
            models_list = self.client.models.list()
            available_models = {m.name.replace("models/", "") for m in models_list}

            for preferred_model in self.MODELS_PREFERRED:
                clean_pref = preferred_model.replace("models/", "")
                if clean_pref in available_models:
                    return clean_pref

            flash_models = [m for m in available_models if "flash" in m.lower()]
            if flash_models:
                return flash_models[0]

            if available_models:
                return next(iter(available_models))

        except Exception as e:
            print(f"Waarschuwing bij ophalen modellenlijst ({e}). Gebruikt standaardmodel.")

        return "gemini-2.5-flash"

    def generate_response(
        self, 
        query: str, 
        prompt_text: str, 
        enable_serp: bool = True
    ) -> Tuple[str, List[str]]:
        """
        Genereert een antwoord op basis van de opgegeven query en context.
        
        Returns:
            Tuple[str, List[str]]: (antwoord_tekst, lijst_van_webbronnen)
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not prompt_text or not prompt_text.strip():
            raise ValueError("Prompt context cannot be empty.")

        # Stel het prompt samen
        full_prompt = f"""Context:
        {prompt_text}

         Question:{query}
         Answer:"""

        # Configuratie opbouwen
        tools = [types.Tool(google_search=types.GoogleSearch())] if enable_serp else []
        
        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            tools=tools,
            temperature=0.3 if not enable_serp else 0.7,
        )

        # Stel fallbacks samen
        fallback_models = [self.model] + [
            m.replace("models/", "") for m in self.MODELS_PREFERRED if m.replace("models/", "") != self.model
        ]

        response = None
        last_exception = None

        # Probeer modellen achter elkaar bij fouten/quota limits
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
                last_exception = e
                print(f"Fout bij model {current_model}: {e}. Probeer volgend model...")

        if not response:
            raise RuntimeError(f"Kan geen antwoord genereren via Gemini API: {last_exception}")

        answer = response.text or ""
        web_sources: List[str] = []

        # Extraheer grounding bronnen (SERP)
        if enable_serp and response.candidates:
            grounding_meta = getattr(response.candidates[0], "grounding_metadata", None)
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

    def ask_with_context(
        self, 
        query: str, 
        context: str, 
        enable_serp: bool = True
    ) -> str:
        '''
        Vereenvoudigde interface om een vraag te stellen met context.
         '''
        
        answer, sources = self.generate_response(query, context, enable_serp=enable_serp)
        
        return answer