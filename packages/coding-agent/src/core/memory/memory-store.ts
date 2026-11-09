// In-memory and local filesystem persistent memory store for conversation segments.

import { randomUUID } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { extractDefaultCues } from "./cue-extractor.ts";
import { SparseIndex } from "./sparse-index.ts";
import type {
	ArchiveSegmentInput,
	HistoryChainOptions,
	MemoryNode,
	MemorySearchResult,
	MemoryStoreOptions,
	SearchByCueOptions,
} from "./types.ts";

function cleanSessionId(idOrPath?: string): string | undefined {
	if (!idOrPath) return undefined;
	if (!idOrPath.includes("/") && !idOrPath.includes("\\") && !idOrPath.endsWith(".jsonl")) {
		return idOrPath;
	}
	const base =
		idOrPath
			.split(/[/\\]/)
			.pop()
			?.replace(/\.jsonl$/, "") ?? idOrPath;
	const underscoreIndex = base.indexOf("_");
	if (underscoreIndex !== -1 && underscoreIndex < base.length - 1) {
		return base.slice(underscoreIndex + 1);
	}
	return base;
}

/** Storage and sparse index manager for conversation turns and history chains. */
export class MemoryStore {
	private readonly nodes: Map<string, MemoryNode> = new Map();
	private readonly sessionNodes: Map<string, string[]> = new Map();
	private readonly index: SparseIndex = new SparseIndex();
	private readonly storageFilePath: string | undefined;
	private loaded = false;

	constructor(options?: MemoryStoreOptions) {
		this.storageFilePath = resolveStoragePath(options);
	}

	private ensureLoaded(): void {
		if (this.loaded) return;
		this.loaded = true;
		if (this.storageFilePath && existsSync(this.storageFilePath)) {
			this.loadFromFile(this.storageFilePath);
		}
	}

	/** Create a purely in-memory store without disk persistence. */
	static inMemory(): MemoryStore {
		return new MemoryStore();
	}

	/** Open or create a memory store backed by a local file. */
	static open(options: MemoryStoreOptions): MemoryStore {
		return new MemoryStore(options);
	}

	/** Archive a conversation segment with its retrieval cues and lossless messages. */
	archiveSessionSegment(input: ArchiveSegmentInput): MemoryNode {
		this.ensureLoaded();

		// Deduplication: prevent duplicate archival of the same session span on retries.
		if (input.firstKeptEntryId) {
			const existingSessionNodeIds = this.sessionNodes.get(input.sessionId);
			if (existingSessionNodeIds) {
				for (const existingId of existingSessionNodeIds) {
					const existingNode = this.nodes.get(existingId);
					if (existingNode && existingNode.firstKeptEntryId === input.firstKeptEntryId) {
						return existingNode;
					}
				}
			}
		}

		let cues: string[];
		if (input.cues !== undefined) {
			// The caller decided the cues, including the decision to have none: a cue-less node
			// stays reachable through its Archived Session Pointer (search must be able to fail).
			cues = input.cues.map((cue) => cue.trim()).filter((cue) => cue.length > 0);
		} else {
			cues = extractDefaultCues({
				messages: input.messages,
				summary: input.summary,
				metadata: input.metadata,
			});
			if (cues.length === 0) {
				throw new Error(
					"Retrieval cues are mandatory when archiving a session segment without explicit cues. None were extracted.",
				);
			}
		}

		const id = input.id ?? randomUUID();
		if (this.nodes.has(id)) {
			throw new Error(`MemoryNode with id "${id}" already exists`);
		}

		const sessionList = this.sessionNodes.get(input.sessionId);
		const autoPrevious = sessionList && sessionList.length > 0 ? sessionList[sessionList.length - 1] : undefined;
		const previousNodeId = input.previousNodeId !== undefined ? input.previousNodeId : autoPrevious;
		const parentSessionId = cleanSessionId(input.parentSessionId);

		const node: MemoryNode = {
			id,
			sessionId: input.sessionId,
			parentSessionId,
			previousNodeId,
			firstKeptEntryId: input.firstKeptEntryId,
			timestamp: input.timestamp ?? Date.now(),
			cues: Array.from(new Set(cues)),
			summary: input.summary,
			messages: [...input.messages],
			metadata: input.metadata ? { ...input.metadata } : undefined,
		};

		this.nodes.set(id, node);
		this.recordSessionNode(node.sessionId, id);
		this.index.addNode(node);

		if (this.storageFilePath) {
			this.appendToFile(node);
		}

		return node;
	}

	/** Search archived nodes by retrieval cue using strict sparse matching. */
	searchByCue(query: string, options?: SearchByCueOptions): MemorySearchResult[] {
		this.ensureLoaded();
		return this.index.search(query, options, (id) => this.nodes.get(id));
	}

	/** Retrieve a memory node by its unique identifier. */
	getNodeById(id: string): MemoryNode | undefined {
		this.ensureLoaded();
		return this.nodes.get(id);
	}

	/** Trace the ancestry chain leading up to the given node. */
	getHistoryChain(nodeId: string, options?: HistoryChainOptions): MemoryNode[] {
		this.ensureLoaded();
		const chain: MemoryNode[] = [];
		const visited = new Set<string>();
		const maxDepth = options?.maxDepth !== undefined && options.maxDepth > 0 ? options.maxDepth : 10;
		let currentId: string | undefined = nodeId;

		while (currentId && chain.length < maxDepth && !visited.has(currentId)) {
			visited.add(currentId);
			const node = this.nodes.get(currentId);
			if (!node) break;
			chain.push(node);

			if (node.previousNodeId) {
				currentId = node.previousNodeId;
			} else if (node.parentSessionId) {
				const parentNodes =
					this.sessionNodes.get(node.parentSessionId) ??
					this.sessionNodes.get(cleanSessionId(node.parentSessionId)!);
				currentId = parentNodes && parentNodes.length > 0 ? parentNodes[parentNodes.length - 1] : undefined;
			} else {
				currentId = undefined;
			}
		}

		const direction = options?.direction ?? "root-to-leaf";
		return direction === "leaf-to-root" ? chain : chain.reverse();
	}

	/** Get all archived nodes belonging to a specific session in arrival order. */
	getNodesBySessionId(sessionId: string): MemoryNode[] {
		this.ensureLoaded();
		const ids = this.sessionNodes.get(sessionId);
		if (!ids) return [];
		const result: MemoryNode[] = [];
		for (const id of ids) {
			const node = this.nodes.get(id);
			if (node) result.push(node);
		}
		return result;
	}

	/** Get all archived memory nodes. */
	getAllNodes(): MemoryNode[] {
		this.ensureLoaded();
		return Array.from(this.nodes.values());
	}

	/** Delete a memory node and update indices without truncating shared file. */
	deleteNode(id: string): boolean {
		this.ensureLoaded();
		const node = this.nodes.get(id);
		if (!node) return false;

		this.removeNodeFromMemory(id);

		if (this.storageFilePath) {
			this.appendTombstone(id);
		}

		return true;
	}

	/** Number of memory nodes stored. */
	size(): number {
		this.ensureLoaded();
		return this.nodes.size;
	}

	/** Clear all stored memory nodes and index entries without truncating shared file. */
	clear(): void {
		this.loaded = true;
		this.nodes.clear();
		this.sessionNodes.clear();
		this.index.clear();

		if (this.storageFilePath) {
			this.appendClear();
		}
	}

	private recordSessionNode(sessionId: string, nodeId: string): void {
		let list = this.sessionNodes.get(sessionId);
		if (!list) {
			list = [];
			this.sessionNodes.set(sessionId, list);
		}
		if (!list.includes(nodeId)) {
			list.push(nodeId);
		}
	}

	private removeNodeFromMemory(id: string): void {
		const node = this.nodes.get(id);
		if (!node) return;
		this.nodes.delete(id);
		this.index.removeNode(id);

		const sessionList = this.sessionNodes.get(node.sessionId);
		if (sessionList) {
			const nextList = sessionList.filter((item) => item !== id);
			if (nextList.length > 0) {
				this.sessionNodes.set(node.sessionId, nextList);
			} else {
				this.sessionNodes.delete(node.sessionId);
			}
		}
	}

	private loadFromFile(filePath: string): void {
		const content = readFileSync(filePath, "utf-8");
		const lines = content.split("\n");

		for (const line of lines) {
			const trimmed = line.trim();
			if (trimmed.length === 0) continue;

			try {
				const parsed = JSON.parse(trimmed) as (MemoryNode & { type?: string }) | { type: string; id?: string };
				if (parsed.type === "clear") {
					this.nodes.clear();
					this.sessionNodes.clear();
					this.index.clear();
					continue;
				}
				if (parsed.type === "delete" && typeof parsed.id === "string") {
					this.removeNodeFromMemory(parsed.id);
					continue;
				}
				const nodeCandidate = parsed as MemoryNode;
				if (
					nodeCandidate.id &&
					nodeCandidate.sessionId &&
					Array.isArray(nodeCandidate.cues) &&
					Array.isArray(nodeCandidate.messages)
				) {
					const node: MemoryNode = {
						...nodeCandidate,
						parentSessionId: cleanSessionId(nodeCandidate.parentSessionId),
					};
					this.nodes.set(node.id, node);
					this.recordSessionNode(node.sessionId, node.id);
					this.index.addNode(node);
				}
			} catch (err) {
				console.warn(
					`[MemoryStore] Warning: skipped malformed JSONL entry in ${filePath}:`,
					err instanceof Error ? err.message : String(err),
				);
			}
		}
	}

	private appendToFile(node: MemoryNode): void {
		if (!this.storageFilePath) return;
		const dir = dirname(this.storageFilePath);
		if (!existsSync(dir)) {
			mkdirSync(dir, { recursive: true });
		}
		appendFileSync(this.storageFilePath, `${JSON.stringify(node)}\n`, "utf-8");
	}

	private appendTombstone(id: string): void {
		if (!this.storageFilePath) return;
		const dir = dirname(this.storageFilePath);
		if (!existsSync(dir)) {
			mkdirSync(dir, { recursive: true });
		}
		appendFileSync(this.storageFilePath, `${JSON.stringify({ type: "delete", id })}\n`, "utf-8");
	}

	private appendClear(): void {
		if (!this.storageFilePath) return;
		const dir = dirname(this.storageFilePath);
		if (!existsSync(dir)) {
			mkdirSync(dir, { recursive: true });
		}
		appendFileSync(this.storageFilePath, `${JSON.stringify({ type: "clear" })}\n`, "utf-8");
	}
}

function resolveStoragePath(options?: MemoryStoreOptions): string | undefined {
	if (options?.storageFilePath) {
		return options.storageFilePath;
	}
	if (options?.storageDir) {
		return join(options.storageDir, "memory-store.jsonl");
	}
	return undefined;
}
