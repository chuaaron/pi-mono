// Public API for dark-layer memory store and sparse retrieval index.

export { type ExtractCuesOptions, extractDefaultCues, parseRetrievalCuesFromSummary } from "./cue-extractor.ts";
export { MemoryStore } from "./memory-store.ts";
export { normalizeCue, SparseIndex, tokenizeText } from "./sparse-index.ts";
export type {
	ArchivedSessionNode,
	ArchiveSegmentInput,
	ChainDirection,
	HistoryChainOptions,
	MemoryNode,
	MemorySearchResult,
	MemoryStoreOptions,
	SearchByCueOptions,
	SparseMatchMode,
} from "./types.ts";
