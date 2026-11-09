import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { parseRetrievalCuesFromSummary } from "../src/core/memory/cue-extractor.ts";
import { MemoryStore } from "../src/core/memory/memory-store.ts";
import { normalizeCue, tokenizeText } from "../src/core/memory/sparse-index.ts";

const testTempDir = join(tmpdir(), `pi-memory-store-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);

beforeAll(() => {
	mkdirSync(testTempDir, { recursive: true });
});

afterAll(() => {
	if (existsSync(testTempDir)) {
		rmSync(testTempDir, { recursive: true, force: true });
	}
});

function createUserMessage(text: string): AgentMessage {
	return {
		role: "user",
		content: text,
		timestamp: Date.now(),
	};
}

function createAssistantToolMessage(): AgentMessage {
	return {
		role: "assistant",
		content: [
			{
				type: "text",
				text: "Inspecting configuration",
			},
			{
				type: "toolCall",
				id: "call_123",
				name: "read_file",
				arguments: { path: "src/auth-config.ts" },
			},
		],
		api: "openai-completions",
		provider: "openai",
		model: "gpt-4",
		usage: {
			input: 10,
			output: 20,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 30,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "toolUse",
		timestamp: Date.now(),
	};
}

describe("MemoryStore - archiving and cues", () => {
	it("archives a session segment with explicit cues and lossless messages", () => {
		const store = MemoryStore.inMemory();
		const userMsg = createUserMessage("How do I migrate the database schema?");
		const assistantMsg = createAssistantToolMessage();

		const node = store.archiveSessionSegment({
			sessionId: "session-1",
			cues: ["database-migration", "schema-change", "sql"],
			summary: "Discussion on migration plan",
			messages: [userMsg, assistantMsg],
			metadata: { env: "staging", priority: "high" },
		});

		expect(node.id).toBeDefined();
		expect(node.sessionId).toBe("session-1");
		expect(node.cues).toEqual(["database-migration", "schema-change", "sql"]);
		expect(node.summary).toBe("Discussion on migration plan");
		expect(node.messages).toHaveLength(2);
		expect(node.messages[0]).toEqual(userMsg);
		expect(node.messages[1]).toEqual(assistantMsg);
		expect(node.metadata).toEqual({ env: "staging", priority: "high" });

		expect(store.getNodeById(node.id)).toEqual(node);
		expect(store.size()).toBe(1);
	});

	it("honors an explicit empty cue list without falling back to extraction", () => {
		const store = MemoryStore.inMemory();

		const node = store.archiveSessionSegment({
			sessionId: "session-empty",
			cues: [],
			messages: [],
		});

		expect(node.cues).toEqual([]);
	});

	it("enforces cue requirement when cues are omitted and extraction yields nothing", () => {
		const store = MemoryStore.inMemory();

		expect(() => {
			store.archiveSessionSegment({
				sessionId: "session-nothing",
				messages: [],
			});
		}).toThrow("Retrieval cues are mandatory");
	});

	it("auto-extracts default cues when explicit cues are omitted", () => {
		const store = MemoryStore.inMemory();
		const userMsg = createUserMessage("Please review docker deployment configuration");

		const node = store.archiveSessionSegment({
			sessionId: "session-auto",
			summary: "Reviewed `docker-compose` settings",
			messages: [userMsg],
		});

		expect(node.cues.length).toBeGreaterThan(0);
		expect(node.cues.some((c) => c.includes("docker"))).toBe(true);
	});

	it("extracts model-judged retrieval cues from ## Retrieval Cues section in summary", () => {
		const store = MemoryStore.inMemory();
		const summary = `## Goal
Implement session archival

## Progress
### Done
- Added memory-store

## Retrieval Cues
- compaction-archival
- dark-layer-memory
- session-tree-pointer
`;
		const node = store.archiveSessionSegment({
			sessionId: "session-model-cues",
			summary,
			messages: [createUserMessage("Save this context")],
		});

		expect(node.cues).toEqual(["compaction-archival", "dark-layer-memory", "session-tree-pointer"]);
	});

	it("parses diverse LLM output formats for ## Retrieval Cues", () => {
		const commaList = `## Goal\nTest\n## Retrieval Cues\ncompaction-watermark, sparse-search, agent-session\n## Next Steps\nDone`;
		expect(parseRetrievalCuesFromSummary(commaList)).toEqual([
			"compaction-watermark",
			"sparse-search",
			"agent-session",
		]);

		const numberedList = `## Summary\nText\n### Retrieval Cues\n1. \`error-recovery\`\n2. \`silent-overflow\`\n`;
		expect(parseRetrievalCuesFromSummary(numberedList)).toEqual(["error-recovery", "silent-overflow"]);

		const placeholder = `## Retrieval Cues\n[3 to 8 concise, specific search keys]\n(none)`;
		expect(parseRetrievalCuesFromSummary(placeholder)).toEqual([]);
	});

	it("automatically links sequential segments within the same session", () => {
		const store = MemoryStore.inMemory();

		const seg1 = store.archiveSessionSegment({
			sessionId: "session-chain",
			cues: ["part-1"],
			messages: [createUserMessage("Turn 1")],
		});

		const seg2 = store.archiveSessionSegment({
			sessionId: "session-chain",
			cues: ["part-2"],
			messages: [createUserMessage("Turn 2")],
		});

		expect(seg1.previousNodeId).toBeUndefined();
		expect(seg2.previousNodeId).toBe(seg1.id);
	});

	it("allows overriding previousNodeId explicitly", () => {
		const store = MemoryStore.inMemory();

		const seg1 = store.archiveSessionSegment({
			id: "node-custom-1",
			sessionId: "session-a",
			cues: ["init"],
			messages: [createUserMessage("Init")],
		});

		const seg2 = store.archiveSessionSegment({
			id: "node-custom-2",
			sessionId: "session-b",
			previousNodeId: seg1.id,
			cues: ["branch"],
			messages: [createUserMessage("Branch point")],
		});

		expect(seg2.previousNodeId).toBe("node-custom-1");
	});

	it("rejects duplicate node ID", () => {
		const store = MemoryStore.inMemory();
		store.archiveSessionSegment({
			id: "node-duplicate",
			sessionId: "sess-1",
			cues: ["cue-1"],
			messages: [createUserMessage("Hello")],
		});

		expect(() => {
			store.archiveSessionSegment({
				id: "node-duplicate",
				sessionId: "sess-2",
				cues: ["cue-2"],
				messages: [createUserMessage("World")],
			});
		}).toThrow('MemoryNode with id "node-duplicate" already exists');
	});
});

describe("MemoryStore - sparse keyword search", () => {
	it("returns strictly empty results when cue does not match", () => {
		const store = MemoryStore.inMemory();
		store.archiveSessionSegment({
			sessionId: "session-1",
			cues: ["docker", "container", "compose"],
			messages: [createUserMessage("Docker setup")],
		});

		const results = store.searchByCue("kubernetes-cluster");
		expect(results).toEqual([]);
	});

	it("returns empty results for empty or whitespace query", () => {
		const store = MemoryStore.inMemory();
		store.archiveSessionSegment({
			sessionId: "session-1",
			cues: ["docker"],
			messages: [createUserMessage("Docker")],
		});

		expect(store.searchByCue("")).toEqual([]);
		expect(store.searchByCue("   ")).toEqual([]);
	});

	it("performs exact cue matching with mode: exact", () => {
		const store = MemoryStore.inMemory();
		const node1 = store.archiveSessionSegment({
			sessionId: "sess-1",
			cues: ["postgres-migration", "schema-v2"],
			messages: [createUserMessage("Postgres migration")],
		});
		store.archiveSessionSegment({
			sessionId: "sess-2",
			cues: ["postgres-query-tuning"],
			messages: [createUserMessage("Postgres tuning")],
		});

		const exactMatches = store.searchByCue("postgres-migration", { mode: "exact" });
		expect(exactMatches).toHaveLength(1);
		expect(exactMatches[0].node.id).toBe(node1.id);
		expect(exactMatches[0].matchedCues).toContain("postgres-migration");
		expect(exactMatches[0].messages).toEqual(node1.messages);

		// With exact matching, a partial token should return strictly empty
		const partial = store.searchByCue("postgres", { exact: true });
		expect(partial).toEqual([]);
	});

	it("performs token-based sparse search with any-token mode", () => {
		const store = MemoryStore.inMemory();
		const node = store.archiveSessionSegment({
			sessionId: "sess-1",
			cues: ["user-authentication-oauth2", "google-login"],
			messages: [createUserMessage("OAuth implementation")],
		});

		const results = store.searchByCue("authentication oauth");
		expect(results).toHaveLength(1);
		expect(results[0].node.id).toBe(node.id);
		expect(results[0].matchedCues).toContain("user-authentication-oauth2");
	});

	it("supports all-tokens mode requiring every query token to match", () => {
		const store = MemoryStore.inMemory();
		store.archiveSessionSegment({
			sessionId: "sess-1",
			cues: ["docker-build", "linux"],
			messages: [createUserMessage("Docker linux")],
		});
		const node2 = store.archiveSessionSegment({
			sessionId: "sess-2",
			cues: ["docker-compose", "production", "linux"],
			messages: [createUserMessage("Docker compose prod")],
		});

		const results = store.searchByCue("docker compose", { mode: "all-tokens" });
		expect(results).toHaveLength(1);
		expect(results[0].node.id).toBe(node2.id);

		// Query with a missing token returns empty
		const missing = store.searchByCue("docker compose kubernetes", { mode: "all-tokens" });
		expect(missing).toEqual([]);
	});

	it("filters search results by sessionId and applies limit", () => {
		const store = MemoryStore.inMemory();
		store.archiveSessionSegment({
			sessionId: "sess-a",
			cues: ["deploy-script"],
			messages: [createUserMessage("Deploy A")],
		});
		const nodeB = store.archiveSessionSegment({
			sessionId: "sess-b",
			cues: ["deploy-script"],
			messages: [createUserMessage("Deploy B")],
		});

		const filtered = store.searchByCue("deploy", { sessionId: "sess-b" });
		expect(filtered).toHaveLength(1);
		expect(filtered[0].node.id).toBe(nodeB.id);

		const limited = store.searchByCue("deploy", { limit: 1 });
		expect(limited).toHaveLength(1);
	});

	it("ranks exact cue matches higher than token matches", () => {
		const store = MemoryStore.inMemory();
		const tokenMatchNode = store.archiveSessionSegment({
			sessionId: "sess-1",
			cues: ["quick-backup-restore-job"],
			messages: [createUserMessage("Backup job")],
		});
		const exactMatchNode = store.archiveSessionSegment({
			sessionId: "sess-2",
			cues: ["backup"],
			messages: [createUserMessage("Backup")],
		});

		const results = store.searchByCue("backup");
		expect(results).toHaveLength(2);
		expect(results[0].node.id).toBe(exactMatchNode.id);
		expect(results[1].node.id).toBe(tokenMatchNode.id);
		expect(results[0].score).toBeGreaterThan(results[1].score);
	});
});

describe("MemoryStore - history chain traversal", () => {
	it("traces a linear chain of nodes via previousNodeId", () => {
		const store = MemoryStore.inMemory();
		store.archiveSessionSegment({
			id: "n1",
			sessionId: "session-chain",
			cues: ["step-1"],
			messages: [createUserMessage("1")],
		});
		store.archiveSessionSegment({
			id: "n2",
			sessionId: "session-chain",
			cues: ["step-2"],
			messages: [createUserMessage("2")],
		});
		store.archiveSessionSegment({
			id: "n3",
			sessionId: "session-chain",
			cues: ["step-3"],
			messages: [createUserMessage("3")],
		});

		const forwardChain = store.getHistoryChain("n3");
		expect(forwardChain.map((n) => n.id)).toEqual(["n1", "n2", "n3"]);

		const reverseChain = store.getHistoryChain("n3", { direction: "leaf-to-root" });
		expect(reverseChain.map((n) => n.id)).toEqual(["n3", "n2", "n1"]);

		const limitedChain = store.getHistoryChain("n3", { maxDepth: 2, direction: "leaf-to-root" });
		expect(limitedChain.map((n) => n.id)).toEqual(["n3", "n2"]);
	});

	it("traverses across sessions via parentSessionId pointer", () => {
		const store = MemoryStore.inMemory();
		const rootNode = store.archiveSessionSegment({
			id: "root-node",
			sessionId: "parent-session",
			cues: ["parent-step"],
			messages: [createUserMessage("Parent")],
		});

		const childNode = store.archiveSessionSegment({
			id: "child-node",
			sessionId: "child-session",
			parentSessionId: "parent-session",
			cues: ["child-step"],
			messages: [createUserMessage("Child")],
		});

		const chain = store.getHistoryChain(childNode.id);
		expect(chain.map((n) => n.id)).toEqual([rootNode.id, childNode.id]);
	});

	it("prevents infinite loops when cyclic pointers exist", () => {
		const store = MemoryStore.inMemory();
		store.archiveSessionSegment({
			id: "cycle-1",
			sessionId: "cycle",
			previousNodeId: "cycle-2",
			cues: ["c1"],
			messages: [createUserMessage("C1")],
		});
		store.archiveSessionSegment({
			id: "cycle-2",
			sessionId: "cycle",
			previousNodeId: "cycle-1",
			cues: ["c2"],
			messages: [createUserMessage("C2")],
		});

		const chain = store.getHistoryChain("cycle-1");
		expect(chain.length).toBeLessThanOrEqual(2);
	});

	it("returns empty array when target node is not found", () => {
		const store = MemoryStore.inMemory();
		expect(store.getHistoryChain("non-existent")).toEqual([]);
	});
});

describe("MemoryStore - local file persistence", () => {
	it("persists segments to disk and reloads them seamlessly", () => {
		const storageFilePath = join(testTempDir, "persisted-store.jsonl");
		const store = MemoryStore.open({ storageFilePath });

		const msg1 = createUserMessage("Initial command");
		store.archiveSessionSegment({
			id: "p-seg-1",
			sessionId: "persisted-sess",
			cues: ["init-setup"],
			messages: [msg1],
		});

		const msg2 = createUserMessage("Followup command");
		store.archiveSessionSegment({
			id: "p-seg-2",
			sessionId: "persisted-sess",
			cues: ["followup-step"],
			messages: [msg2],
		});

		expect(existsSync(storageFilePath)).toBe(true);

		// Reload store from the same file
		const reloaded = MemoryStore.open({ storageFilePath });
		expect(reloaded.size()).toBe(2);

		const loadedSeg1 = reloaded.getNodeById("p-seg-1");
		expect(loadedSeg1).toBeDefined();
		expect(loadedSeg1?.messages).toEqual([msg1]);

		const loadedSeg2 = reloaded.getNodeById("p-seg-2");
		expect(loadedSeg2?.previousNodeId).toBe("p-seg-1");

		// Search operates on reloaded index
		const searchResults = reloaded.searchByCue("init-setup");
		expect(searchResults).toHaveLength(1);
		expect(searchResults[0].node.id).toBe("p-seg-1");

		// Chain works on reloaded store
		const chain = reloaded.getHistoryChain("p-seg-2");
		expect(chain.map((n) => n.id)).toEqual(["p-seg-1", "p-seg-2"]);
	});

	it("handles malformed JSON lines gracefully during load", () => {
		const corruptedFile = join(testTempDir, "corrupted.jsonl");
		const validNode = {
			id: "valid-1",
			sessionId: "sess-valid",
			cues: ["valid-cue"],
			messages: [{ role: "user", content: "ok", timestamp: 123 }],
			timestamp: 123,
		};
		writeFileSync(corruptedFile, `${JSON.stringify(validNode)}\nINVALID_JSON_CORRUPT\n`, "utf-8");

		const store = MemoryStore.open({ storageFilePath: corruptedFile });
		expect(store.size()).toBe(1);
		expect(store.getNodeById("valid-1")).toBeDefined();
	});

	it("updates persistent file when deleting a node", () => {
		const storageFilePath = join(testTempDir, "delete-store.jsonl");
		const store = MemoryStore.open({ storageFilePath });

		store.archiveSessionSegment({
			id: "del-1",
			sessionId: "del-sess",
			cues: ["keep"],
			messages: [createUserMessage("keep")],
		});
		store.archiveSessionSegment({
			id: "del-2",
			sessionId: "del-sess",
			cues: ["remove"],
			messages: [createUserMessage("remove")],
		});

		expect(store.size()).toBe(2);
		const deleted = store.deleteNode("del-2");
		expect(deleted).toBe(true);
		expect(store.size()).toBe(1);

		const reloaded = MemoryStore.open({ storageFilePath });
		expect(reloaded.size()).toBe(1);
		expect(reloaded.getNodeById("del-2")).toBeUndefined();
		expect(reloaded.searchByCue("remove")).toEqual([]);
	});
});

describe("sparse-index utilities", () => {
	it("normalizes cue strings", () => {
		expect(normalizeCue("  Postgres-DB  ")).toBe("postgres-db");
	});

	it("tokenizes text with Latin and CJK terms", () => {
		const tokens = tokenizeText("docker-compose v2 部署测试");
		expect(tokens).toContain("docker");
		expect(tokens).toContain("compose");
		expect(tokens).toContain("v2");
		expect(tokens).toContain("部署测试");
	});
});
