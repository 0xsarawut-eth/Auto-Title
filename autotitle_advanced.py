# autotitle_advanced.py
# Advanced Intelligent Title Generator – NLP Project
# @author: ALI GHANBARI | GitHub: https://github.com/aligh993/Auto-Title
# @email: alighanbari446@gmail.com

import re
import spacy
import yake
import numpy as np
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Load models (lazy initialization)
_nlp = None
_yake_kw_extractor = None
_sentence_model = None
_bart_summarizer = None
_t5_generator = None

def _load_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

def _load_yake():
    global _yake_kw_extractor
    if _yake_kw_extractor is None:
        _yake_kw_extractor = yake.KeywordExtractor(lan="en", n=3, top=10)
    return _yake_kw_extractor

def _load_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _sentence_model

def _load_bart():
    global _bart_summarizer
    if _bart_summarizer is None:
        _bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return _bart_summarizer

def _load_t5():
    global _t5_generator
    if _t5_generator is None:
        _t5_generator = pipeline("text2text-generation", model="t5-small")
    return _t5_generator


class AutoTitle:
    def __init__(self):
        print("Loading models... (this may take a moment)")
        _load_nlp()
        _load_yake()
        _load_sentence_model()
        _load_bart()
        _load_t5()
        print("AutoTitle ready!")

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;\:\-\(\)\[\]\{\}\'\"\\\/]', '', text)
        return text.strip()

    def _extractive_title(self, text: str) -> Tuple[str, float]:
        """Use YAKE to extract keyphrases and form a title."""
        kw_extractor = _load_yake()
        keywords = kw_extractor.extract_keywords(text)
        if not keywords:
            return "Summary", 0.3

        # Take top 1-3 keywords and form title
        top_phrases = [kw for kw, score in keywords[:3]]
        candidate = " ".join(top_phrases).strip()
        candidate = candidate.title()

        # Truncate if too long
        words = candidate.split()
        if len(words) > 12:
            candidate = " ".join(words[:12]) + "..."

        return candidate, 1.0 - keywords[0][1]  # inverse score

    def _abstractive_title(self, text: str) -> Tuple[str, float]:
        """Use BART → T5 pipeline for abstractive title."""
        bart = _load_bart()
        t5 = _load_t5()

        try:
            # Step 1: Summarize with BART
            summary = bart(
                text,
                max_length=60,
                min_length=20,
                do_sample=False,
                truncation=True
            )[0]['summary_text']

            # Step 2: Generate title with T5
            prompt = f"generate title: {summary}"
            result = t5(prompt, max_length=20, do_sample=False)[0]['generated_text']
            return result.strip(), 0.9
        except Exception as e:
            return "Generated Title", 0.4

    def _postprocess(self, title: str, max_words: int = 12) -> str:
        """Enforce rules: length, capitalization, remove quotes, filter clickbait."""
        if not title.strip():
            return "Untitled Document"

        # Remove quotes
        title = re.sub(r'^["\']|["\']$', '', title.strip())
        title = re.sub(r'^\[|\]$', '', title)

        # Title case
        title = title.title()

        # Truncate
        words = title.split()
        if len(words) > max_words:
            title = " ".join(words[:max_words]) + "…"

        # Clickbait filter
        clickbait_patterns = [
            r'\bYou Won\'?t Believe\b',
            r'\bShocking\b',
            r'\bSecret\b.*\bRevealed\b',
            r'\b\d+ Ways?\b',
            r'\bWhy You Should\b'
        ]
        for pattern in clickbait_patterns:
            if re.search(pattern, title, re.I):
                title = re.sub(pattern, '', title, flags=re.I).strip()
                title = title or "Key Insights"

        # Final cleanup
        title = re.sub(r'\s+', ' ', title)
        return title.strip()

    def _score_title(self, title: str, text: str) -> float:
        """Compute semantic similarity between title and lead sentences."""
        model = _load_sentence_model()
        doc = _load_nlp()(text)
        sentences = [sent.text for sent in doc.sents][:3]  # first 3 sentences
        if not sentences:
            return 0.5

        lead = " ".join(sentences)
        emb1 = model.encode(title, convert_to_tensor=True)
        emb2 = model.encode(lead, convert_to_tensor=True)
        return util.cos_sim(emb1, emb2).item()

    def generate(
        self,
        text: str,
        max_words: int = 12,
        prefer_abstractive: bool = True
    ) -> dict:
        """
        Generate the best title with metadata.
        Returns: {'title': str, 'method': str, 'confidence': float, 'word_count': int}
        """
        if not text or len(text.strip()) < 20:
            return {
                "title": "Short Text",
                "method": "fallback",
                "confidence": 0.1,
                "word_count": 2
            }

        text = self._clean_text(text)

        candidates = []

        # 1. Abstractive candidate
        if prefer_abstractive and len(text.split()) > 50:
            abs_title, abs_conf = self._abstractive_title(text)
            abs_title = self._postprocess(abs_title, max_words)
            sim_score = self._score_title(abs_title, text)
            candidates.append({
                "title": abs_title,
                "method": "abstractive",
                "base_conf": abs_conf,
                "sim_score": sim_score,
                "final_conf": abs_conf * 0.6 + sim_score * 0.4
            })

        # 2. Extractive candidate
        ext_title, ext_conf = self._extractive_title(text)
        ext_title = self._postprocess(ext_title, max_words)
        sim_score = self._score_title(ext_title, text)
        candidates.append({
            "title": ext_title,
            "method": "extractive",
            "base_conf": ext_conf,
            "sim_score": sim_score,
            "final_conf": ext_conf * 0.5 + sim_score * 0.5
        })

        # Select best
        best = max(candidates, key=lambda x: x["final_conf"])
        word_count = len(best["title"].split())

        return {
            "title": best["title"],
            "method": best["method"],
            "confidence": round(best["final_conf"], 3),
            "word_count": word_count
        }


# ===============================
# EXAMPLE USAGE
# ===============================

if __name__ == "__main__":
    # Initialize
    generator = AutoTitle()

    # Sample long text (article)
    sample_text = """
    Scientists at MIT have developed a new artificial intelligence system that can 
    predict the onset of Alzheimer's disease up to six years before symptoms appear. 
    The model analyzes speech patterns, writing style, and subtle cognitive markers 
    in everyday language to detect early signs of neurodegeneration. In clinical 
    trials involving 500 patients, the system achieved 92% accuracy in identifying 
    individuals who would later develop the condition. This breakthrough could 
    revolutionize early intervention strategies and dramatically improve patient 
    outcomes. The research was published today in Nature Medicine.
    """

    # Generate title
    result = generator.generate(sample_text, max_words=10)

    print("\n" + "="*60)
    print(" AUTO TITLE GENERATION RESULT ")
    print("="*60)
    print(f"Title:       {result['title']}")
    print(f"Method:      {result['method'].title()}")
    print(f"Confidence:  {result['confidence']:.3f}")
    print(f"Words:       {result['word_count']}")
    print("="*60)

    # Batch example
    print("\nBatch Processing Example:")
    texts = [
        "Global temperatures reached record highs in 2024, with July being the hottest month ever recorded.",
        "A new species of deep-sea fish was discovered near the Mariana Trench.",
        "Tesla announced a breakthrough in solid-state battery technology."
    ]

    for i, text in enumerate(texts, 1):
        res = generator.generate(text, max_words=8)
        print(f"{i}. {res['title']} ({res['method']})")