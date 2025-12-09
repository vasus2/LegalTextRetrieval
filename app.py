"""
Briefly - Legal Precedent Search UI

A clean, professional Streamlit interface for searching legal case precedents.
This is a CS410 course project prototype demonstrating BM25-based retrieval.

Author: Briefly Team
"""

import streamlit as st
import logging
import sys
import os
from typing import List, Optional
from retrieval.bm25_retriever import BM25Retriever, SearchResult

# Add briefly directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'briefly'))

try:
    from dense_retrieval import DenseRetriever
    DENSE_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    DENSE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Dense retrieval not available: {e}")

try:
    from hybrid_retrieval import HybridRetriever
    HYBRID_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    HYBRID_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Hybrid retrieval not available: {e}")

try:
    from citation_graph import CitationGraph
    CITATION_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    CITATION_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Citation graph not available: {e}")

# ============================================================
# Configuration
# ============================================================

CORPUS_PATH = "bm25-files/docs00.json"
METADATA_PATH = "data/case_data.json"
DEFAULT_TOP_K = 10
MAX_TOP_K = 50

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Briefly – Legal Precedent Search",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# Minimal, Clean Styling
# ============================================================

st.markdown("""
<style>
    /* Hide deploy button and streamlit branding */
    [data-testid="stToolbar"] {
        display: none;
    }
    
    /* Light cream background */
    [data-testid="stAppViewContainer"] {
        background-color: #faf8f3;
    }
    
    [data-testid="stHeader"] {
        background-color: #faf8f3;
    }
    
    /* All text and titles black */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stText {
        color: #000000 !important;
    }
    
    /* Force all titles to be black */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    
    /* All other text elements */
    p, span, div, label, caption {
        color: #000000 !important;
    }
    
    /* Captions and small text */
    .stCaption {
        color: #000000 !important;
    }
    
    /* Expander content */
    .streamlit-expanderContent {
        color: #000000 !important;
    }
    
    .streamlit-expanderContent p, .streamlit-expanderContent div {
        color: #000000 !important;
    }
    
    /* Input fields */
    .stTextInput input {
        font-size: 1rem;
        background-color: #ffffff;
        color: #000000;
        border: 1px solid #d1d5db;
    }
    
    .stTextInput input::placeholder {
        color: #9ca3af;
    }
    
    .stTextInput label {
        color: #000000 !important;
    }
    
    /* Number input */
    .stNumberInput input {
        background-color: #ffffff;
        color: #000000;
    }
    
    .stNumberInput label {
        color: #000000 !important;
    }
    
    /* Selectbox (dropdown) */
    .stSelectbox select {
        background-color: #ffffff;
        color: #000000;
    }
    
    .stSelectbox label {
        color: #000000 !important;
    }
    
    /* Selectbox internal components */
    .stSelectbox [data-baseweb="select"] {
        background-color: #ffffff !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Selectbox selected value */
    .stSelectbox [data-baseweb="select"] div[class*="st-"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Selectbox input */
    .stSelectbox input {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Selectbox dropdown menu when opened */
    [data-baseweb="popover"] {
        background-color: #ffffff !important;
    }
    
    [role="listbox"] {
        background-color: #ffffff !important;
    }
    
    [role="option"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    [role="option"]:hover {
        background-color: #f3f4f6 !important;
        color: #000000 !important;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #2563eb;
        color: #ffffff;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f3f1eb;
        color: #000000 !important;
    }
    
    /* Expander when expanded - subtle highlight */
    details[open] > summary {
        background-color: #ebe8df !important;
    }
    
    /* Expander content area */
    .streamlit-expanderContent {
        background-color: #f8f6f0;
    }
    
    /* Text area - all states */
    .stTextArea textarea {
        background-color: #ffffff;
        color: #000000 !important;
    }
    
    /* Text area disabled state */
    .stTextArea textarea:disabled {
        color: #000000 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* Target Streamlit's internal textarea classes */
    textarea[class*="st-"] {
        color: #000000 !important;
    }
    
    textarea[disabled] {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* Base input wrapper */
    [data-baseweb="base-input"] textarea {
        color: #000000 !important;
    }
    
    .stTextArea label {
        color: #000000 !important;
    }
    
    /* Info/warning boxes */
    .stAlert {
        color: #000000 !important;
    }
    
    /* Divider */
    hr {
        border-color: #d1d5db;
    }
    
    /* Consistent spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #faf8f3;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Backend Integration
# ============================================================

@st.cache_resource
def load_retriever() -> Optional[BM25Retriever]:
    """
    Load and cache the BM25 retriever.
    
    This function is cached so the BM25 index is built only once
    and reused across all queries, improving performance.
    
    Returns:
        BM25Retriever instance, or None if loading fails
    """
    try:
        retriever = BM25Retriever(
            corpus_path=CORPUS_PATH,
            metadata_path=METADATA_PATH
        )
        logger.info(f"Loaded retriever with {retriever.get_corpus_size()} documents")
        return retriever
    except FileNotFoundError as e:
        logger.error(f"Corpus file not found: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading retriever: {e}")
        return None


@st.cache_resource
def load_dense_retriever():
    """Load and cache the dense retriever."""
    if not DENSE_AVAILABLE:
        return None
    try:
        return DenseRetriever()
    except Exception as e:
        logger.error(f"Error loading dense retriever: {e}")
        return None

@st.cache_resource
def load_hybrid_retriever():
    """Load and cache the hybrid retriever."""
    if not HYBRID_AVAILABLE:
        return None
    try:
        return HybridRetriever(alpha=0.6)  # 60% BM25, 40% dense
    except Exception as e:
        logger.error(f"Error loading hybrid retriever: {e}")
        return None

@st.cache_resource
def load_citation_graph():
    """Load and cache the citation graph."""
    if not CITATION_AVAILABLE:
        return None
    try:
        return CitationGraph()
    except Exception as e:
        logger.error(f"Error loading citation graph: {e}")
        return None

def search_cases(query: str, top_k: int = 10, method: str = "bm25") -> List[dict]:
    """
    Search for legal cases using specified retrieval method.
    
    Args:
        query: Natural language search query
        top_k: Number of results to return
        method: Retrieval method ("bm25", "dense", or "hybrid")
        
    Returns:
        List of result dictionaries with standardized fields
    """
    if method == "bm25":
        retriever = load_retriever()
        if not retriever:
            return []
        
        # Get SearchResult objects from retriever
        results = retriever.search(query, top_k=top_k)
        
        # Convert to standardized dict format for UI
        return [
            {
                'rank': r.rank,
                'case_id': r.doc_id,
                'title': r.case_name or f"Case {r.doc_id}",
                'citation': r.cite,
                'court': r.court,
                'date': r.date,
                'score': r.score,
                'score_type': 'BM25',
                'snippet': r.snippet,
                'full_text': r.full_text
            }
            for r in results
        ]
    
    elif method == "dense":
        retriever = load_dense_retriever()
        if not retriever:
            st.error("Dense retrieval not available. Run build_dense_embeddings.py first.")
            return []
        
        results = retriever.search(query, top_k=top_k)
        
        # Standardize field names to match BM25 format
        standardized = []
        for r in results:
            standardized.append({
                'rank': r['rank'],
                'case_id': r['case_id'],
                'title': r.get('name', f"Case {r['case_id']}"),
                'citation': r.get('cite', None),
                'court': r.get('court', None),
                'date': r.get('date', None),
                'score': r['score'],
                'score_type': 'Semantic',
                'snippet': r.get('snippet', ''),
                'full_text': r.get('snippet', '')
            })
        
        return standardized
    
    elif method == "hybrid":
        retriever = load_hybrid_retriever()
        if not retriever:
            st.error("Hybrid retrieval not available. Run build_dense_embeddings.py first.")
            return []
        
        results = retriever.search(query, top_k=top_k)
        
        # Standardize field names to match BM25 format
        standardized = []
        for r in results:
            standardized.append({
                'rank': r['rank'],
                'case_id': r['case_id'],
                'title': r.get('name', f"Case {r['case_id']}"),
                'citation': r.get('cite', None),
                'court': r.get('court', None),
                'date': r.get('date', None),
                'score': r['hybrid_score'],
                'score_type': 'Hybrid',
                'bm25_score': r.get('bm25_score', 0),
                'dense_score': r.get('dense_score', 0),
                'snippet': r.get('snippet', ''),
                'full_text': r.get('snippet', '')
            })
        
        return standardized
    
    else:
        raise ValueError(f"Unknown retrieval method: {method}")

# ============================================================
# UI Components
# ============================================================

def render_header():
    """Render the page header with title and subtitle."""
    st.title("Briefly – Legal Precedent Search")
    st.markdown(
        "A CS410 course project prototype for searching legal case precedents using BM25 retrieval."
    )
    st.divider()


def render_corpus_info():
    """Display information about the loaded corpus."""
    retriever = load_retriever()
    
    if not retriever:
        st.error("**Error:** Could not load the case corpus.")
        st.info(f"Please ensure the corpus file exists at: `{CORPUS_PATH}`")
        st.info("Run `python briefly/bm25_pipeline.py` to generate the corpus.")
        st.stop()
    
    corpus_size = retriever.get_corpus_size()
    st.info(f"**Corpus loaded:** {corpus_size:,} legal case documents indexed and ready for search")


def render_search_form() -> tuple[str, int, bool, str]:
    """
    Render the search input form.
    
    Returns:
        Tuple of (query, top_k, search_clicked, method)
    """
    with st.container():
        st.subheader("Search Query")
        
        # Query input
        query = st.text_input(
            "Enter your legal query",
            placeholder="e.g., contract breach damages, due process rights, negligence liability...",
            help="Enter a natural language query to search for relevant legal cases"
        )
        
        # Controls row
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_clicked = st.button(
                "Search",
                type="primary",
                use_container_width=True
            )
        
        with col2:
            # Method selector
            method_options = ["BM25"]
            if DENSE_AVAILABLE:
                method_options.append("Dense")
            if HYBRID_AVAILABLE:
                method_options.append("Hybrid")
            
            method = st.selectbox(
                "Method",
                options=method_options,
                help="BM25: keyword matching | Dense: semantic similarity | Hybrid: combination"
            )
        
        with col3:
            top_k = st.number_input(
                "Top results",
                min_value=1,
                max_value=MAX_TOP_K,
                value=DEFAULT_TOP_K,
                step=1,
                help=f"Number of results to display (1-{MAX_TOP_K})"
            )
        
        st.divider()
    
    return query, top_k, search_clicked, method.lower()


def render_result_card(result: dict):
    """
    Render a single search result as a clean card.
    
    Args:
        result: Dictionary containing result data
    """
    with st.container():
        # Rank badge and title
        col1, col2 = st.columns([1, 11])
        
        with col1:
            st.markdown(f"### #{result['rank']}")
        
        with col2:
            st.markdown(f"### {result['title']}")
        
        # Metadata row with icons
        metadata_parts = []
        if result['citation']:
            metadata_parts.append(f"Citation: {result['citation']}")
        if result['court']:
            metadata_parts.append(f"Court: {result['court']}")
        if result['date']:
            metadata_parts.append(f"Date: {result['date']}")
        
        if metadata_parts:
            st.caption(" • ".join(metadata_parts))
        
        # Score badge - show appropriate score type
        score_type = result.get('score_type', 'BM25')
        score_value = result['score']
        
        if score_type == 'Hybrid':
            # Show all three scores for hybrid
            st.markdown(
                f"**Hybrid Score:** `{score_value:.3f}` | "
                f"BM25: `{result.get('bm25_score', 0):.2f}` | "
                f"Dense: `{result.get('dense_score', 0):.3f}`"
            )
        else:
            st.markdown(
                f"**{score_type} Score:** `{score_value:.3f}`"
            )
        
        # Citation metrics (if available)
        citation_graph = load_citation_graph()
        if citation_graph and citation_graph.case_exists(result['case_id']):
            metrics = citation_graph.get_citation_metrics(result['case_id'])
            citation_parts = []
            
            if metrics['citation_count'] > 0:
                citation_parts.append(f"📚 Cited by **{metrics['citation_count']}** cases")
            if metrics['cites_count'] > 0:
                citation_parts.append(f"🔗 Cites **{metrics['cites_count']}** cases")
            
            if citation_parts:
                st.caption(" • ".join(citation_parts))
        
        # Snippet
        st.markdown("**Preview:**")
        st.markdown(f"> {result['snippet']}")
        
        # Citation exploration buttons
        citation_graph = load_citation_graph()
        if citation_graph and citation_graph.case_exists(result['case_id']):
            metrics = citation_graph.get_citation_metrics(result['case_id'])
            
            if metrics['citation_count'] > 0 or metrics['cites_count'] > 0:
                st.markdown("**Explore Citations:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    if metrics['citation_count'] > 0:
                        if st.button(
                            f"📚 Show {metrics['citation_count']} citing cases",
                            key=f"citing_{result['case_id']}_{result['rank']}",
                            use_container_width=True
                        ):
                            # Store citation exploration request in session state
                            st.session_state['citation_explore'] = {
                                'type': 'citing',
                                'case_id': result['case_id'],
                                'case_name': result['title']
                            }
                            st.rerun()
                
                with col2:
                    if metrics['cites_count'] > 0:
                        if st.button(
                            f"🔗 Show {metrics['cites_count']} cited cases",
                            key=f"cited_{result['case_id']}_{result['rank']}",
                            use_container_width=True
                        ):
                            # Store citation exploration request in session state
                            st.session_state['citation_explore'] = {
                                'type': 'cited',
                                'case_id': result['case_id'],
                                'case_name': result['title']
                            }
                            st.rerun()
        
        # Expandable full text
        if result['full_text']:
            with st.expander("View full case text"):
                st.text_area(
                    "Full Text",
                    value=result['full_text'],
                    height=300,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"fulltext_{result['case_id']}_{result['rank']}"
                )
        
        st.divider()


def render_results(results: List[dict], query: str):
    """
    Render the search results section.
    
    Args:
        results: List of result dictionaries
        query: The search query that produced these results
    """
    if not results:
        st.info("No cases found for this query. Try reformulating your search with different terms.")
        return
    
    # Results header
    st.subheader(f"Search Results ({len(results)} cases)")
    st.markdown(f"**Query:** *{query}*")
    st.divider()
    
    # Render each result
    for result in results:
        render_result_card(result)


def render_about_section():
    """Render the about/info section at the bottom."""
    st.divider()
    
    with st.expander("ℹ️ About this prototype"):
        st.markdown("""
        ### Briefly – Legal Precedent Retrieval System
        
        This is a prototype legal case search system built as a CS410 course project.
        
        **Current Features:**
        - BM25-based retrieval over ~5,000 legal cases from the LePaRD dataset
        - Natural language query interface
        - Ranked results with case metadata and text snippets
        
        **Planned Enhancements:**
        - 🔄 Multiple retrieval methods (BM25, Embedding-based, Hybrid)
        - 🔗 Citation-based recommendations
        - 📊 Advanced filtering by court, date, jurisdiction
        - 💾 Query history and saved searches
        - 📈 Relevance feedback and personalization
        
        **Technology Stack:**
        - Python, Streamlit, rank_bm25, HuggingFace datasets
        - Data: LePaRD (Legal Precedent Retrieval Dataset)
        """)

# ============================================================
# Main Application
# ============================================================

def main():
    """Main application entry point."""
    
    # Render header
    render_header()
    
    # Show corpus info (or error if corpus not loaded)
    render_corpus_info()
    
    # Render search form
    query, top_k, search_clicked, method = render_search_form()
    
    # Handle citation exploration (if triggered from a result card)
    if 'citation_explore' in st.session_state:
        explore_data = st.session_state['citation_explore']
        citation_graph = load_citation_graph()
        
        if citation_graph:
            st.info(f"**Citation Exploration:** {explore_data['case_name']}")
            
            if explore_data['type'] == 'citing':
                # Show cases that cite this case
                citing_cases = citation_graph.get_citing_cases(explore_data['case_id'])
                st.subheader(f"Cases Citing This Case ({len(citing_cases)})")
                
                if citing_cases:
                    for i, case in enumerate(citing_cases, 1):
                        st.markdown(f"**{i}. {case['name']}**")
                        st.caption(f"Citation: {case['cite']} • Cited by {case['citation_count']} cases")
                        st.divider()
                else:
                    st.info("No citing cases found in the corpus.")
            
            elif explore_data['type'] == 'cited':
                # Show cases cited by this case
                cited_cases = citation_graph.get_cited_cases(explore_data['case_id'])
                st.subheader(f"Cases Cited by This Case ({len(cited_cases)})")
                
                if cited_cases:
                    for i, case in enumerate(cited_cases, 1):
                        st.markdown(f"**{i}. {case['name']}**")
                        st.caption(f"Citation: {case['cite']} • Cited by {case['citation_count']} cases")
                        st.divider()
                else:
                    st.info("No cited cases found in the corpus.")
            
            # Clear the exploration state
            if st.button("← Back to search"):
                del st.session_state['citation_explore']
                st.rerun()
        
        st.stop()
    
    # Handle search
    if search_clicked:
        # Validate input
        if not query or not query.strip():
            st.warning("Please enter a search query.")
            st.stop()
        
        # Perform search with spinner
        with st.spinner(f"Searching legal precedents using {method.upper()} method..."):
            try:
                results = search_cases(query.strip(), top_k=top_k, method=method)
            except Exception as e:
                logger.exception("Search failed")
                st.error(f"An error occurred during search: {str(e)}")
                st.exception(e)
                st.stop()
        
        # Display results
        render_results(results, query)


if __name__ == "__main__":
    main()
