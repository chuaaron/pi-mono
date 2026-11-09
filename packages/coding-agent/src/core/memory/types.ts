// Core type definitions for dark-layer memory storage and sparse retrieval indexing.

import type { AgentMessage } from "@earendil-works/pi-agent-core";

/** Mode of matching query terms against stored retrieval cues. */
export type SparseMatchMode = "exact" | "all-tokens" | "any-token";

/** Traversal direction for history chain reconstruction. */
export type ChainDirection = "root-to-leaf" | "leaf-to-root";

/** Archived memory segment node holding lossless messages and graph pointers. */
export interface MemoryNode {
	/** Unique identifier of the memory node. */
	id: string;
	/** Session identifier that this segment belongs to. */
	sessionId: string;
	/** Optional parent session identifier for session-level lineage. */
	parentSessionId?: string;
	/** Optional entry ID where the uncompacted context begins, used for retry deduplication. */
	firstKeptEntryId?: string;
	/** Optional pointer to the immediately preceding memory node in the history chain. */
	previousNodeId?: string;
	/** Unix timestamp in milliseconds when this node was archived. */
	timestamp: number;
	/** Non-empty collection of retrieval cues and keywords. */
	cues: string[];
	/** Optional concise summary of the archived conversation segment. */
	summary?: string;
	/** Lossless message sequence preserved in original structure. */
	messages: AgentMessage[];
	/** Optional contextual metadata. */
	metadata?: Record<string, unknown>;
}

/** Alias for MemoryNode representing an archived session node. */
export type ArchivedSessionNode = MemoryNode;

/** Input parameters for archiving a session segment into the memory store. */
export interface ArchiveSegmentInput {
	/** Optional explicit node identifier. Defaults to an auto-generated unique ID. */
	id?: string;
	/** Session identifier that this segment belongs to. */
	sessionId: string;
	/** Optional parent session identifier for session lineage. */
	parentSessionId?: string;
	/** Optional entry ID where the uncompacted context begins, used for retry deduplication. */
	firstKeptEntryId?: string;
	/** Optional pointer to the preceding memory node. */
	previousNodeId?: string;
	/**
	 * Retrieval cues. When omitted, cues are extracted automatically from the segment.
	 * An explicit empty array is honored as-is: the node is archived without cues.
	 */
	cues?: string[];
	/** Optional concise summary of the segment. */
	summary?: string;
	/** Lossless sequence of turns and messages. */
	messages: AgentMessage[];
	/** Creation timestamp in milliseconds. Defaults to Date.now(). */
	timestamp?: number;
	/** Additional contextual metadata. */
	metadata?: Record<string, unknown>;
}

/** Search parameters for cue-based sparse retrieval. */
export interface SearchByCueOptions {
	/** Matching mode: "exact", "all-tokens", or "any-token". Defaults to "any-token". */
	mode?: SparseMatchMode;
	/** Convenience flag. When true, enforces mode="exact". */
	exact?: boolean;
	/** Maximum number of results to return. If omitted or non-positive, returns all matches. */
	limit?: number;
	/** Filter matches to a specific session. */
	sessionId?: string;
}

/** Result item returned by cue search containing the matched node and its messages. */
export interface MemorySearchResult {
	/** The matched memory node. */
	node: MemoryNode;
	/** Direct access to the node's lossless messages. */
	messages: AgentMessage[];
	/** List of cues on this node that matched the query. */
	matchedCues: string[];
	/** Relevance ranking score computed from cue and token matches. */
	score: number;
}

/** Options for tracing history chains across memory nodes. */
export interface HistoryChainOptions {
	/** Maximum depth of preceding nodes to traverse. Defaults to 10. */
	maxDepth?: number;
	/** Order of returned chain: root-to-leaf or leaf-to-root. Defaults to root-to-leaf. */
	direction?: ChainDirection;
}

/** Configuration options for initializing a MemoryStore instance. */
export interface MemoryStoreOptions {
	/** Directory path to persist memory entries as JSONL. If omitted, operates in memory only. */
	storageDir?: string;
	/** Explicit file path for persistence. Overrides storageDir if provided. */
	storageFilePath?: string;
}
