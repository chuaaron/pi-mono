// Inverted index and sparse keyword retrieval matching for memory cues.

import type { MemoryNode, MemorySearchResult, SearchByCueOptions, SparseMatchMode } from "./types.ts";

/** Inverted sparse index providing deterministic keyword and token matching across memory cues. */
export class SparseIndex {
	private readonly exactIndex: Map<string, Set<string>> = new Map();
	private readonly tokenIndex: Map<string, Set<string>> = new Map();
	private readonly nodeCues: Map<string, string[]> = new Map();
	private readonly nodeTokens: Map<string, Set<string>> = new Map();

	/** Add or re-index a memory node's retrieval cues. */
	addNode(node: MemoryNode): void {
		this.removeNode(node.id);

		const normalizedCues = node.cues.map(normalizeCue).filter((c) => c.length > 0);
		this.nodeCues.set(node.id, normalizedCues);

		const allTokens = new Set<string>();
		for (const cue of normalizedCues) {
			let exactSet = this.exactIndex.get(cue);
			if (!exactSet) {
				exactSet = new Set();
				this.exactIndex.set(cue, exactSet);
			}
			exactSet.add(node.id);

			const tokens = tokenizeText(cue);
			for (const token of tokens) {
				allTokens.add(token);
				let tokenSet = this.tokenIndex.get(token);
				if (!tokenSet) {
					tokenSet = new Set();
					this.tokenIndex.set(token, tokenSet);
				}
				tokenSet.add(node.id);
			}
		}

		this.nodeTokens.set(node.id, allTokens);
	}

	/** Remove a memory node from the inverted index. */
	removeNode(nodeId: string): void {
		const existingCues = this.nodeCues.get(nodeId);
		if (existingCues) {
			for (const cue of existingCues) {
				const exactSet = this.exactIndex.get(cue);
				if (exactSet) {
					exactSet.delete(nodeId);
					if (exactSet.size === 0) {
						this.exactIndex.delete(cue);
					}
				}
			}
			this.nodeCues.delete(nodeId);
		}

		const existingTokens = this.nodeTokens.get(nodeId);
		if (existingTokens) {
			for (const token of existingTokens) {
				const tokenSet = this.tokenIndex.get(token);
				if (tokenSet) {
					tokenSet.delete(nodeId);
					if (tokenSet.size === 0) {
						this.tokenIndex.delete(token);
					}
				}
			}
			this.nodeTokens.delete(nodeId);
		}
	}

	/** Search nodes matching the query cue under specified options. Returns empty array on no match. */
	search(
		query: string,
		options: SearchByCueOptions | undefined,
		nodeResolver: (id: string) => MemoryNode | undefined,
	): MemorySearchResult[] {
		const trimmedQuery = query.trim();
		if (trimmedQuery.length === 0) {
			return [];
		}

		const isExact = options?.exact === true || options?.mode === "exact";
		const mode: SparseMatchMode = isExact ? "exact" : (options?.mode ?? "any-token");

		if (mode === "exact") {
			return this.searchExact(trimmedQuery, options?.sessionId, options?.limit, nodeResolver);
		}

		if (mode === "all-tokens") {
			return this.searchAllTokens(trimmedQuery, options?.sessionId, options?.limit, nodeResolver);
		}

		return this.searchAnyToken(trimmedQuery, options?.sessionId, options?.limit, nodeResolver);
	}

	/** Clear all indexed cues and tokens. */
	clear(): void {
		this.exactIndex.clear();
		this.tokenIndex.clear();
		this.nodeCues.clear();
		this.nodeTokens.clear();
	}

	private searchExact(
		query: string,
		sessionId: string | undefined,
		limit: number | undefined,
		nodeResolver: (id: string) => MemoryNode | undefined,
	): MemorySearchResult[] {
		const normalized = normalizeCue(query);
		const ids = this.exactIndex.get(normalized);
		if (!ids || ids.size === 0) {
			return [];
		}

		const results: MemorySearchResult[] = [];
		for (const id of ids) {
			const node = nodeResolver(id);
			if (!node) continue;
			if (sessionId && node.sessionId !== sessionId) continue;

			const matchedCues = node.cues.filter((c) => normalizeCue(c) === normalized);
			results.push({
				node,
				messages: node.messages,
				matchedCues,
				score: EXACT_MATCH_SCORE * matchedCues.length,
			});
		}

		return rankAndLimit(results, limit);
	}

	private searchAllTokens(
		query: string,
		sessionId: string | undefined,
		limit: number | undefined,
		nodeResolver: (id: string) => MemoryNode | undefined,
	): MemorySearchResult[] {
		const queryTokens = tokenizeText(query);
		if (queryTokens.length === 0) {
			return [];
		}

		let candidateIds: Set<string> | undefined;
		for (const token of queryTokens) {
			const matching = this.tokenIndex.get(token);
			if (!matching || matching.size === 0) {
				return [];
			}
			if (!candidateIds) {
				candidateIds = new Set(matching);
			} else {
				for (const id of candidateIds) {
					if (!matching.has(id)) {
						candidateIds.delete(id);
					}
				}
				if (candidateIds.size === 0) {
					return [];
				}
			}
		}

		if (!candidateIds || candidateIds.size === 0) {
			return [];
		}

		const results: MemorySearchResult[] = [];
		const normalizedQuery = normalizeCue(query);

		for (const id of candidateIds) {
			const node = nodeResolver(id);
			if (!node) continue;
			if (sessionId && node.sessionId !== sessionId) continue;

			const matchedCues: string[] = [];
			let score = 0;

			for (const cue of node.cues) {
				const normCue = normalizeCue(cue);
				const cueTokens = tokenizeText(cue);
				const overlap = queryTokens.filter((qt) => cueTokens.includes(qt));
				if (overlap.length > 0) {
					matchedCues.push(cue);
					score += overlap.length * TOKEN_MATCH_SCORE;
				}
				if (normCue === normalizedQuery) {
					score += EXACT_MATCH_SCORE;
				}
			}

			if (matchedCues.length > 0) {
				results.push({
					node,
					messages: node.messages,
					matchedCues: deduplicateStrings(matchedCues),
					score,
				});
			}
		}

		return rankAndLimit(results, limit);
	}

	private searchAnyToken(
		query: string,
		sessionId: string | undefined,
		limit: number | undefined,
		nodeResolver: (id: string) => MemoryNode | undefined,
	): MemorySearchResult[] {
		const normalizedQuery = normalizeCue(query);
		const queryTokens = tokenizeText(query);
		const candidateIds = new Set<string>();

		const exactMatches = this.exactIndex.get(normalizedQuery);
		if (exactMatches) {
			for (const id of exactMatches) candidateIds.add(id);
		}

		for (const token of queryTokens) {
			const tokenMatches = this.tokenIndex.get(token);
			if (tokenMatches) {
				for (const id of tokenMatches) candidateIds.add(id);
			}
		}

		if (candidateIds.size === 0) {
			return [];
		}

		const results: MemorySearchResult[] = [];
		for (const id of candidateIds) {
			const node = nodeResolver(id);
			if (!node) continue;
			if (sessionId && node.sessionId !== sessionId) continue;

			const matchedCues: string[] = [];
			let score = 0;

			for (const cue of node.cues) {
				const normCue = normalizeCue(cue);
				if (normCue === normalizedQuery) {
					score += EXACT_MATCH_SCORE;
					matchedCues.push(cue);
					continue;
				}
				if (normCue.includes(normalizedQuery) || normalizedQuery.includes(normCue)) {
					score += SUBSTRING_MATCH_SCORE;
					matchedCues.push(cue);
					continue;
				}
				const cueTokens = tokenizeText(cue);
				const overlap = queryTokens.filter((qt) => cueTokens.includes(qt));
				if (overlap.length > 0) {
					score += overlap.length * TOKEN_MATCH_SCORE;
					matchedCues.push(cue);
				}
			}

			if (score > 0 && matchedCues.length > 0) {
				results.push({
					node,
					messages: node.messages,
					matchedCues: deduplicateStrings(matchedCues),
					score,
				});
			}
		}

		return rankAndLimit(results, limit);
	}
}

const EXACT_MATCH_SCORE = 10;
const SUBSTRING_MATCH_SCORE = 5;
const TOKEN_MATCH_SCORE = 1;

/** Normalize a retrieval cue by trimming and converting to lower case. */
export function normalizeCue(cue: string): string {
	return cue.trim().toLowerCase();
}

/** Tokenize text into search terms supporting alphanumeric words and CJK characters. */
export function tokenizeText(text: string): string[] {
	const normalized = text.toLowerCase();
	const matches = normalized.match(/[\p{L}\p{N}]+/gu);
	if (!matches) {
		return [];
	}
	return deduplicateStrings(matches);
}

function deduplicateStrings(items: string[]): string[] {
	return Array.from(new Set(items));
}

function rankAndLimit(results: MemorySearchResult[], limit: number | undefined): MemorySearchResult[] {
	results.sort((a, b) => {
		if (b.score !== a.score) {
			return b.score - a.score;
		}
		return b.node.timestamp - a.node.timestamp;
	});

	if (limit !== undefined && limit > 0) {
		return results.slice(0, limit);
	}
	return results;
}
