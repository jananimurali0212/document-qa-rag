"""
Answer Generator Module.

Invokes Google Gemini API or an offline deterministic fallback generator to produce
evidence-grounded answers with source citations, strictly abstaining when evidence is insufficient.
"""

from __future__ import annotations

import os
import re
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from src.retrieval.vector_store import SearchResult
from src.generation.prompt_builder import PromptBuilder, GroundedPrompt


@dataclass
class Citation:
    """Source reference supporting the answer."""
    document_name: str
    page_number: int
    chunk_id: str
    snippet: str
    score: float


@dataclass
class AnswerResult:
    """Complete generation outcome."""
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[SearchResult]
    model_name: str
    latency_ms: float
    grounded: bool
    abstained: bool


class AnswerGenerator:
    """
    Modular LLM answer generator.
    Connects to Google Gemini API (google-genai / google.generativeai)
    with a graceful deterministic offline fallback engine for zero-dependency execution.
    """

    DEFAULT_MODEL = "gemini-3.8-flash"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        prompt_builder: Optional[PromptBuilder] = None
    ):
        """
        Initialize the Answer Generator.

        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY environment variable).
            model_name: Google Gemini model identifier.
            temperature: Sampling temperature (low temperature = high grounding).
            prompt_builder: PromptBuilder instance.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name
        self.temperature = temperature
        self.prompt_builder = prompt_builder or PromptBuilder()

        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize Google Gemini client if API key is present."""
        if not self.api_key:
            return

        # Attempt modern google.genai or google.generativeai
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            self._client_type = "google-genai"
            return
        except Exception:
            pass

        try:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=self.api_key)
            self._client = legacy_genai.GenerativeModel(self.model_name)
            self._client_type = "google-generativeai"
            return
        except Exception:
            self._client = None

    def set_api_key(self, api_key: str) -> None:
        """Update API key dynamically (e.g. from Streamlit sidebar)."""
        self.api_key = api_key.strip()
        self._init_client()

    STOP_WORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
        "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for",
        "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
        "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
        "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
        "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
        "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
        "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
        "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
        "why", "will", "with", "would", "you", "your", "yours", "yourself", "yourselves",
        "explain", "describe", "define", "give", "write", "state", "list", "name", "tell"
    }

    def _offline_generate(
        self,
        question: str,
        retrieved_chunks: List[SearchResult]
    ) -> str:
        """
        Deterministic, evidence-grounded answer extraction used when offline or without API key.
        Extracts and structures the most relevant factual passages from the retrieved context.
        """
        if not retrieved_chunks:
            return self.prompt_builder.ABSTAIN_PHRASE

        # Extract meaningful content keywords from query
        raw_q_tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", question.lower())
        content_keywords = [w for w in raw_q_tokens if w not in self.STOP_WORDS and len(w) > 1]
        active_keywords = content_keywords if content_keywords else raw_q_tokens

        if not active_keywords:
            return self.prompt_builder.ABSTAIN_PHRASE

        # Score candidate sentences/paragraphs across chunks
        scored_candidates: List[tuple[float, str, SearchResult]] = []

        BOILERPLATE_PATTERNS = [
            r"question bank", r"department of", r"course outcome", r"bloom's level",
            r"part [a-c]", r"marks\)", r"details of the course", r"regulations"
        ]

        for chunk in retrieved_chunks:
            # Partition chunk into paragraphs or sentences
            lines = [l.strip() for l in chunk.text.split("\n") if len(l.strip()) > 15]
            sentences = []
            for line in lines:
                sents = re.split(r"(?<=[.!?])\s+", line)
                sentences.extend([s.strip() for s in sents if len(s.strip()) > 15])

            if not sentences:
                sentences = [chunk.text.strip()]

            for sent in sentences:
                s_lower = sent.lower()

                # Penalize boilerplate / metadata lines
                is_boilerplate = any(re.search(pat, s_lower) for pat in BOILERPLATE_PATTERNS)
                if is_boilerplate and not any(kw in s_lower for kw in active_keywords if kw not in {"python", "course"}):
                    continue

                sent_tokens = set(re.findall(r"\b[a-zA-Z0-9_]+\b", s_lower))
                keyword_hits = []
                for kw in active_keywords:
                    for st in sent_tokens:
                        if kw == st or kw in st or st in kw:
                            keyword_hits.append(kw)
                            break
                        min_l = min(len(kw), len(st))
                        if min_l >= 5 and kw[:min_l - 2] == st[:min_l - 2]:
                            keyword_hits.append(kw)
                            break

                if not keyword_hits:
                    continue

                # Score based on keyword density & exact match count
                score = (len(keyword_hits) / len(active_keywords)) * 2.0

                # Phrase match bonus
                if len(active_keywords) >= 2:
                    for i in range(len(active_keywords) - 1):
                        pair = f"{active_keywords[i]} {active_keywords[i+1]}"
                        if pair in s_lower:
                            score += 1.5

                # Definition / answering pattern bonus
                if any(pat in s_lower for pat in ["is defined as", "refers to", "consists of", "is a", "are used to", "syntax:"]):
                    score += 0.8

                # Question prompt indicator bonus
                if re.match(r"^\d+[\.\)]", sent.strip()):
                    score += 0.5

                scored_candidates.append((score, sent, chunk))

        if not scored_candidates:
            # Fall back to best chunk text if high similarity score
            top_chunk = retrieved_chunks[0]
            if top_chunk.score > 0.4:
                # Return the most substantive line
                clean_lines = [l for l in top_chunk.text.split("\n") if len(l.strip()) > 25 and not any(re.search(p, l.lower()) for p in BOILERPLATE_PATTERNS)]
                excerpt = " ".join(clean_lines[:3]) if clean_lines else top_chunk.text[:250]
                return (
                    f"**From {top_chunk.document_name} (Page {top_chunk.page_number}):**\n\n"
                    f"{excerpt.strip()}\n\n"
                    f"*(Source: Page {top_chunk.page_number}, Chunk `{top_chunk.chunk_id}`)*"
                )
            return self.prompt_builder.ABSTAIN_PHRASE

        # Sort candidate sentences by score descending
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Select top non-overlapping excerpts
        selected_sentences: List[str] = []
        primary_chunk = scored_candidates[0][2]

        for s_score, s_text, c_source in scored_candidates[:4]:
            if not any(s_text in existing or existing in s_text for existing in selected_sentences):
                selected_sentences.append(s_text)

        formatted_answer = "\n\n".join(f"- {s}" if len(selected_sentences) > 1 else s for s in selected_sentences)

        return (
            f"**Answer based on {primary_chunk.document_name} (Page {primary_chunk.page_number}):**\n\n"
            f"{formatted_answer}\n\n"
            f"*(Reference: Page {primary_chunk.page_number}, Chunk `{primary_chunk.chunk_id}`)*"
        )

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[SearchResult]
    ) -> AnswerResult:
        """
        Generate answer from question and retrieved context using Gemini or offline engine.

        Args:
            question: Natural language question.
            retrieved_chunks: Retrieved passages from Retriever.

        Returns:
            AnswerResult containing answer, citations, latency, and status flags.
        """
        start_time = time.perf_counter()
        prompt_data = self.prompt_builder.build_prompt(question, retrieved_chunks)

        # Build citations
        citations = [
            Citation(
                document_name=c.document_name,
                page_number=c.page_number,
                chunk_id=c.chunk_id,
                snippet=c.text[:160] + "..." if len(c.text) > 160 else c.text,
                score=c.score
            )
            for c in retrieved_chunks
        ]

        if not retrieved_chunks:
            latency = (time.perf_counter() - start_time) * 1000.0
            return AnswerResult(
                answer=self.prompt_builder.ABSTAIN_PHRASE,
                citations=[],
                retrieved_chunks=[],
                model_name=self.model_name + " (abstain)",
                latency_ms=round(latency, 2),
                grounded=False,
                abstained=True
            )

        # Attempt Online Gemini Generation if client configured
        if self._client is not None:
            try:
                if self._client_type == "google-genai":
                    response = self._client.models.generate_content(
                        model=self.model_name,
                        contents=prompt_data.user_prompt,
                        config={
                            "system_instruction": prompt_data.system_instruction,
                            "temperature": self.temperature,
                        }
                    )
                    answer_text = response.text or ""
                else:
                    # legacy google.generativeai
                    combined = f"{prompt_data.system_instruction}\n\n{prompt_data.user_prompt}"
                    response = self._client.generate_content(combined)
                    answer_text = response.text or ""

                latency = (time.perf_counter() - start_time) * 1000.0
                abstained = (
                    "insufficient" in answer_text.lower()
                    or "cannot answer" in answer_text.lower()
                    or "not mentioned" in answer_text.lower()
                    or "not contain sufficient" in answer_text.lower()
                )
                return AnswerResult(
                    answer=answer_text.strip(),
                    citations=citations,
                    retrieved_chunks=retrieved_chunks,
                    model_name=f"{self.model_name} (Gemini API)",
                    latency_ms=round(latency, 2),
                    grounded=not abstained,
                    abstained=abstained
                )
            except Exception as e:
                # Fall back to offline generation if API call encounters network/quota error
                offline_text = self._offline_generate(question, retrieved_chunks)
                latency = (time.perf_counter() - start_time) * 1000.0
                abstained = (
                    self.prompt_builder.ABSTAIN_PHRASE in offline_text
                    or "not contain sufficient" in offline_text.lower()
                )
                return AnswerResult(
                    answer=f"{offline_text}\n\n*(Notice: Gemini API fallback due to: {str(e)})*",
                    citations=citations,
                    retrieved_chunks=retrieved_chunks,
                    model_name=f"{self.model_name} (Offline Fallback)",
                    latency_ms=round(latency, 2),
                    grounded=not abstained,
                    abstained=abstained
                )

        # Run Offline Deterministic Generator
        offline_text = self._offline_generate(question, retrieved_chunks)
        latency = (time.perf_counter() - start_time) * 1000.0
        abstained = (
            self.prompt_builder.ABSTAIN_PHRASE in offline_text
            or "not contain sufficient" in offline_text.lower()
        )
        return AnswerResult(
            answer=offline_text,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            model_name="Extractive Grounded Generator (Offline/Local)",
            latency_ms=round(latency, 2),
            grounded=not abstained,
            abstained=abstained
        )
