from langchain_core.prompts import PromptTemplate

score_prompt_template = PromptTemplate(
    input_variables=["query", "summaries_text"],
    template="""You are a precision-oriented information retriever. Your goal is to identify which document contains the specific answer to the User Query.
    
    Scoring Rubric:
    - 90-100: The summary explicitly mentions the core specific entities, dates, or quantitative targets requested in the query.
    - 60-89: The summary covers the exact sub-topic and context but lacks the specific data point or date.
    - 30-59: The summary covers the broad domain but does not address the specific context of the query.
    - 0-29: The summary is unrelated.

    Instructions:
    1. Do not reward "Broad Reviews" if they do not contain the specific details requested.
    2. If a query asks for a "Target" or "Law," look for mentions of "policy," "regulation," or "percentage."
    3. Return ONLY a JSON array: 
    [
    {{"filename": "doc1.pdf", "score": 99}},
    {{"filename": "doc2.pdf", "score": 100}},
    {{"filename": "doc3.pdf", "score": 22}}
    ]
    4. Do NOT include markdown or preamble.

    User Query: 
    {query}

    Summaries:
    {summaries_text}
    """
)

prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="""You are a research assistant chatbot. Answer the user's question using ONLY the context below.
    
    Context:
    {context}
    
    Question:
    {query}

    Instructions:
    - Provide a clear, explanatory, and accurate answer
    - Only use information from the provided context
    - If the context doesn't contain enough information, say so
    - Be specific and cite details when available

    Answer:"""
)

# New prompts for document processing
page_summary_prompt = PromptTemplate(
    input_variables=["page_content"],
    template="""Summarize the following page in 2–3 concise sentences.
    Also if you find any publication year mention it after summary, only year.
    Also, add keywords related to the content of the page at the end of the summary after publication year.
    Example: "This page discusses... Publication Year: 2020. Keywords: physics, quantum mechanics."

    Page:
    {page_content}
    """
)

document_summary_prompt = PromptTemplate(
    input_variables=["summaries"],
    template="""Combine the following page summaries into ONE coherent summary,
    concise document-level summary,
    provide publication year if mentioned.
    also keep keywords at the end.
    Summary need to be plain text.
    
    {summaries}
    """
)

publication_year_prompt = PromptTemplate(
    input_variables=["summary"],
    template="""Extract the publication year from the document summary below.
    If unknown, return 0, if not found provide random year from 2015 to 2024.
    Return ONLY the year as an integer.
    
    Summary:
    {summary}
    """
)

keywords_prompt = PromptTemplate(
    input_variables=["summary"],
    template="""Extract 5–10 concise keywords from the following summary.
    Only Return as a comma-separated list of keywords.

    Summary:
    {summary}
    """
)

normalization_prompt = PromptTemplate(
    input_variables=["page"],
    template="""
    SYSTEM:
    You are a deterministic document normalization engine for a RAG system.
    You strictly follow instructions and NEVER add meta-text, explanations, or commentary.

    USER:
    TASK:
    Normalize the document below into a canonical structure.

    RULES (MANDATORY):
    1. Identify all logical headings (including bold text, numbered lines, or ALL CAPS).
    2. Convert headings to:
    - #  → Document title
    - ## → Main sections
    - ### → Subsections
    3. Preserve ALL text exactly as written.
    - Do NOT paraphrase
    - Do NOT summarize
    - Do NOT remove or add content
    4. Preserve tables (<table id="">) and images (<::) exactly as-is.
    5. Do NOT add markdown fences, explanations, introductions, or conclusions.
    6. Output ONLY the normalized document text.

    OUTPUT CONSTRAINT:
    - The output MUST start with the first header.
    - The output MUST end with the final line of content.
    - Any text outside the document is INVALID.

    CONTENT TO NORMALIZE:
    {page}
    """
)