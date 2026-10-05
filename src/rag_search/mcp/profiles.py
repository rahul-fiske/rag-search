"""Per-host settings for the MCP adapter (no dependency on the `mcp` package)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    client: str          # identity sent to the daemons; selects the collections it may use
    note: str = ""       # extra line for the server instructions


PROFILES = {
    "claude": Profile("claude", "claude"),
}
DEFAULT_PROFILE = "claude"


def get_profile(name: str) -> Profile:
    """A known host's profile, or a plain one for any other name (a new client)."""
    return PROFILES.get(name) or Profile(name, name)

INSTRUCTIONS = """\
Local document search (RAG) over the user's own indexed documents.

Workflow:
1. rag_list_collections - see which collections you may use (and whether anything is
   indexed). Only collections authorised for this host are listed; any other name is
   reported as unknown.
2. rag_search - meaning-based search (BM25 + embeddings + reranker). Results carry the
   source file and PAGE NUMBER; cite them.
3. rag_grep - exact strings / IDs / numbers / regexes when semantic search is the wrong tool.
4. To add documents: put files in a registered source folder, then rag_index_update (a background run in
   the indexer daemon) and poll rag_index_status. New documents become searchable
   automatically when the run finishes. Only one indexing run exists at a time; calling
   rag_index_update again reports the active one, and restart=true starts it over.
A 'warming_up' reply means the search engine is still loading its models: retry shortly.
"""


def instructions(profile: Profile) -> str:
    return INSTRUCTIONS + (("\n" + profile.note + "\n") if profile.note else "")
