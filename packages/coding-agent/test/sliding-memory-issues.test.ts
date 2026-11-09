import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent, type StreamFn } from "@earendil-works/pi-agent-core";
import { createAssistantMessageEventStream, fauxAssistantMessage, type Model } from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AgentSession } from "../src/core/agent-session.ts";
import { AuthStorage } from "../src/core/auth-storage.ts";
import {
	compact,
	formatArchivedSessionPointer,
	getMessagesToArchive,
	prepareCompaction,
} from "../src/core/compaction/index.ts";
import {
	createExtensionRuntime,
	type Extension,
	type SessionBeforeCompactEvent,
} from "../src/core/extensions/index.ts";
import { parseRetrievalCuesFromSummary } from "../src/core/memory/cue-extractor.ts";
import { MemoryStore } from "../src/core/memory/index.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import { createSyntheticSourceInfo } from "../src/core/source-info.ts";
import { createMemorySearchTool } from "../src/core/tools/index.ts";
import { createInMemoryModelRegistry, getModelRuntime } from "./model-runtime-test-utils.ts";
import { createTestResourceLoader } from "./utilities.ts";

const mockModel: Model<"anthropic-messages"> = {
	id: "test-model",
	name: "Test Model",
	api: "anthropic-messages",
	provider: "test-provider",
	baseUrl: "https://api.anthropic.com",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 200_000,
	maxTokens: 4096,
};

function createMockStreamFn(summaryText: string): StreamFn {
	return () => {
		const stream = createAssistantMessageEventStream();
		queueMicrotask(() =>
			stream.push({
				type: "done",
				reason: "stop",
				message: {
					...fauxAssistantMessage(summaryText),
					api: mockModel.api,
					provider: mockModel.provider,
					model: mockModel.id,
					usage: {
						input: 100,
						output: 50,
						cacheRead: 0,
						cacheWrite: 0,
						totalTokens: 150,
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
					},
				},
			}),
		);
		return stream;
	};
}

describe("Sliding memory issues cluster verification", () => {
	let tempDir: string;

	beforeEach(() => {
		tempDir = join(tmpdir(), `pi-test-issues-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		mkdirSync(tempDir, { recursive: true });
	});

	afterEach(() => {
		if (tempDir && existsSync(tempDir)) {
			rmSync(tempDir, { recursive: true, force: true });
		}
	});

	describe("Issue A: Split-turn prefix is archived alongside messagesToSummarize", () => {
		it("archives both messagesToSummarize and turnPrefixMessages into MemoryStore", async () => {
			const sessionManager = SessionManager.inMemory(tempDir);
			const memoryStore = MemoryStore.inMemory();

			sessionManager.appendMessage({
				role: "user",
				content: "Turn 1: user request",
				timestamp: 1000,
			});
			sessionManager.appendMessage({
				...fauxAssistantMessage("Turn 1: assistant response"),
				api: mockModel.api,
				provider: mockModel.provider,
				model: mockModel.id,
				timestamp: 2000,
			});
			sessionManager.appendMessage({
				role: "user",
				content: "Turn 2: user follow up",
				timestamp: 3000,
			});
			sessionManager.appendMessage({
				...fauxAssistantMessage("Turn 2: assistant mid-turn progress A2-1"),
				api: mockModel.api,
				provider: mockModel.provider,
				model: mockModel.id,
				timestamp: 4000,
			});
			sessionManager.appendMessage({
				...fauxAssistantMessage("Turn 2: assistant final progress A2-2"),
				api: mockModel.api,
				provider: mockModel.provider,
				model: mockModel.id,
				timestamp: 5000,
			});

			const preparation = prepareCompaction(sessionManager.getBranch(), {
				enabled: true,
				strategy: "sliding-memory",
				reserveTokens: 1000,
				keepRecentTokens: 1,
			});

			expect(preparation).toBeDefined();
			expect(preparation!.isSplitTurn).toBe(true);
			expect(preparation!.messagesToSummarize.length).toBeGreaterThan(0);
			expect(preparation!.turnPrefixMessages.length).toBeGreaterThan(0);

			const allArchived = getMessagesToArchive(preparation!);
			expect(allArchived.length).toBe(
				preparation!.messagesToSummarize.length + preparation!.turnPrefixMessages.length,
			);

			const result = await compact(
				preparation!,
				mockModel,
				"test-key",
				undefined,
				undefined,
				undefined,
				undefined,
				createMockStreamFn("Summary\n\n## Retrieval Cues\n- topic1\n- topic2"),
				undefined,
				undefined,
				undefined,
				sessionManager.getSessionId(),
				memoryStore,
			);

			const nodeId = result.details?.memoryNodeId;
			expect(nodeId).toBeDefined();
			const node = memoryStore.getNodeById(nodeId!);
			expect(node).toBeDefined();
			expect(node!.messages).toHaveLength(allArchived.length);

			// Both Turn 1 and Turn 2 prefix messages are stored
			const contents = node!.messages.map((m) => {
				const c = (m as Record<string, unknown>).content;
				return typeof c === "string" ? c : JSON.stringify(c);
			});
			expect(contents.some((c) => c.includes("Turn 1: user request"))).toBe(true);
			expect(contents.some((c) => c.includes("Turn 2: assistant mid-turn progress A2-1"))).toBe(true);
		});
	});

	describe("Issue B: memory_search returns bounded results and truncates large messages", () => {
		it("returns summary+cues by default and applies head/tail truncation on large messages", async () => {
			const memoryStore = MemoryStore.inMemory();
			const massiveMessage = `START_${"x".repeat(10000)}_END`;

			memoryStore.archiveSessionSegment({
				id: "node-large",
				sessionId: "session-1",
				cues: ["massive", "payload"],
				summary: "Archived segment with massive payload",
				messages: [{ role: "user", content: massiveMessage, timestamp: 1000 }],
			});

			const tool = createMemorySearchTool(memoryStore);

			// 1. Query search defaults to summary + cues (does NOT dump message)
			const queryRes = await tool.execute("call-1", { query: "massive payload" });
			const queryText = queryRes.content[0].type === "text" ? queryRes.content[0].text : "";
			expect(queryText).toContain("Archived segment with massive payload");
			expect(queryText).toContain('Re-query with nodeId: "node-large" for full message details');
			expect(queryText).not.toContain(massiveMessage);

			// 2. Direct nodeId lookup returns truncated message
			const nodeRes = await tool.execute("call-2", { nodeId: "node-large" });
			const nodeText = nodeRes.content[0].type === "text" ? nodeRes.content[0].text : "";
			expect(nodeText).toContain("START_");
			expect(nodeText).toContain("_END");
			expect(nodeText).toContain("truncated");
			expect(nodeText.length).toBeLessThan(5000);
		});

		it("respects default result limit of 5 on cue search", async () => {
			const memoryStore = MemoryStore.inMemory();
			for (let i = 1; i <= 10; i++) {
				memoryStore.archiveSessionSegment({
					id: `node-${i}`,
					sessionId: "session-multi",
					cues: ["common-cue"],
					summary: `Summary ${i}`,
					messages: [{ role: "user", content: `Message ${i}`, timestamp: i }],
				});
			}

			const tool = createMemorySearchTool(memoryStore);
			const res = await tool.execute("call-limit", { query: "common-cue" });
			expect(res.details?.matchCount).toBe(5);
			expect(res.details?.nodeIds).toHaveLength(5);
		});
	});

	describe("Issue C: Cross-session history chain resolves when parentSession is a file path", () => {
		it("resolves ancestor history chain across session files", () => {
			const parentSessionId = "550e8400-e29b-41d4-a716-446655440000";
			const parentFilePath = `/Users/test/.pi/agent/sessions/2026-11-09T08-00-00-000Z_${parentSessionId}.jsonl`;
			const childSessionId = "660e8400-e29b-41d4-a716-446655440001";

			const memoryStore = MemoryStore.inMemory();

			memoryStore.archiveSessionSegment({
				id: "parent-node-1",
				sessionId: parentSessionId,
				cues: ["parent-feature"],
				summary: "Parent session work",
				messages: [{ role: "user", content: "Parent request", timestamp: 100 }],
			});

			memoryStore.archiveSessionSegment({
				id: "child-node-1",
				sessionId: childSessionId,
				parentSessionId: parentFilePath,
				cues: ["child-feature"],
				summary: "Child session work",
				messages: [{ role: "user", content: "Child request", timestamp: 200 }],
			});

			// History chain traced from child-node-1 must traverse into parent-node-1
			const chain = memoryStore.getHistoryChain("child-node-1", { direction: "root-to-leaf" });
			expect(chain).toHaveLength(2);
			expect(chain[0].id).toBe("parent-node-1");
			expect(chain[1].id).toBe("child-node-1");
		});
	});

	describe("Issue D: File-op XML does not leak into retrieval cues", () => {
		it("terminates cue parsing before <read-files> and <modified-files>", () => {
			const summaryWithFileOps = `This is a summary of the past work.

## Retrieval Cues
- cue-alpha
- cue-beta
- cue-gamma

<read-files>
src/a.ts
src/b.ts
</read-files>
<modified-files>
src/c.ts
</modified-files>`;

			const cues = parseRetrievalCuesFromSummary(summaryWithFileOps);
			expect(cues).toEqual(["cue-alpha", "cue-beta", "cue-gamma"]);
			expect(cues).not.toContain("<read-files>");
			expect(cues).not.toContain("src/a.ts");
		});

		it("caps cues per pointer line in formatArchivedSessionPointer", () => {
			const longCues = Array.from({ length: 30 }, (_, i) => `cue-${i + 1}`);
			const pointer = formatArchivedSessionPointer({
				id: "node-test",
				sessionId: "sess-test",
				cues: longCues,
			});
			const line = pointer.split("\n")[1];
			expect(line.startsWith("Retrieval Cues:")).toBe(true);
			const cuesInLine = line.replace("Retrieval Cues:", "").split(", ");
			expect(cuesInLine.length).toBeLessThanOrEqual(10);
		});
	});

	describe("Issue E: Progress floor prevents compaction thrash", () => {
		it("returns undefined when uncompacted span does not exceed keepRecentTokens budget", () => {
			const sessionManager = SessionManager.inMemory(tempDir);
			sessionManager.appendMessage({
				role: "user",
				content: "A short user message",
				timestamp: 1000,
			});
			sessionManager.appendMessage({
				...fauxAssistantMessage("A short assistant response"),
				api: mockModel.api,
				provider: mockModel.provider,
				model: mockModel.id,
				timestamp: 2000,
			});

			// When keepRecentTokens is 20,000, 2 short messages fit easily within budget
			const prep = prepareCompaction(sessionManager.getBranch(), {
				enabled: true,
				strategy: "sliding-memory",
				reserveTokens: 1000,
				keepRecentTokens: 20000,
			});
			expect(prep).toBeUndefined();
		});

		it("returns undefined when dropped tokens are below progress floor", () => {
			const sessionManager = SessionManager.inMemory(tempDir);
			sessionManager.appendCompaction("Previous summary", "uuid-1", 1000);
			sessionManager.appendMessage({
				role: "user",
				content: "Hi",
				timestamp: 1000,
			});
			sessionManager.appendMessage({
				...fauxAssistantMessage("Hello!"),
				api: mockModel.api,
				provider: mockModel.provider,
				model: mockModel.id,
				timestamp: 2000,
			});
			sessionManager.appendMessage({
				role: "user",
				content: "How are you?",
				timestamp: 3000,
			});
			sessionManager.appendMessage({
				...fauxAssistantMessage("I am doing well!"),
				api: mockModel.api,
				provider: mockModel.provider,
				model: mockModel.id,
				timestamp: 4000,
			});

			const prep = prepareCompaction(sessionManager.getBranch(), {
				enabled: true,
				strategy: "sliding-memory",
				reserveTokens: 1000,
				keepRecentTokens: 200,
			});
			// Dropped tokens (< 500) do not justify compaction overhead
			expect(prep).toBeUndefined();
		});
	});

	describe("Issue F: Extension compaction appends Archived Session Pointer and falls back to cues", async () => {
		it("appends pointer and generates cues when extension provides summary without cues", async () => {
			const authStorage = AuthStorage.inMemory();
			const modelRegistry = await createInMemoryModelRegistry(authStorage);
			const modelRuntime = getModelRuntime(modelRegistry);

			const sessionManager = SessionManager.inMemory(tempDir);
			const settingsManager = SettingsManager.inMemory({
				compaction: {
					enabled: true,
					strategy: "sliding-memory",
					keepRecentTokens: 1,
				},
			});

			sessionManager.appendMessage({
				role: "user",
				content: "Implement feature X",
				timestamp: 1000,
			});
			sessionManager.appendMessage({
				...fauxAssistantMessage("Implemented feature X"),
				api: mockModel.api,
				provider: mockModel.provider,
				model: mockModel.id,
				timestamp: 2000,
			});
			sessionManager.appendMessage({
				role: "user",
				content: "Next turn",
				timestamp: 3000,
			});

			const agent = new Agent({
				getApiKey: () => "test-key",
				streamFn: createMockStreamFn("- feature-x\n- implementation"),
				initialState: {
					model: mockModel,
					systemPrompt: "",
					tools: [],
				},
			});

			const handlers = new Map<string, ((event: any, ctx: any) => Promise<any>)[]>();
			handlers.set("session_before_compact", [
				async (event: SessionBeforeCompactEvent) => {
					return {
						compaction: {
							summary: "Custom extension summary of work.",
							firstKeptEntryId: event.preparation.firstKeptEntryId,
							tokensBefore: event.preparation.tokensBefore,
						},
					};
				},
			]);

			const extension: Extension = {
				path: "test-extension",
				resolvedPath: "/test/test-extension.ts",
				sourceInfo: createSyntheticSourceInfo("<test:test-extension>", { source: "test" }),
				handlers,
				tools: new Map(),
				messageRenderers: new Map(),
				commands: new Map(),
				flags: new Map(),
				shortcuts: new Map(),
			};

			const runtime = createExtensionRuntime();
			const resourceLoader = {
				...createTestResourceLoader(),
				getExtensions: () => ({ extensions: [extension], errors: [], runtime }),
			};

			const session = new AgentSession({
				agent,
				sessionManager,
				settingsManager,
				resourceLoader,
				modelRuntime,
				cwd: tempDir,
			});

			const result = await session.compact();
			expect(result.summary).toContain("Custom extension summary of work.");
			expect(result.summary).toContain("[Archived Session Pointer: Node");
			const details = result.details as Record<string, unknown> | undefined;
			expect(details?.memoryNodeId).toBeDefined();

			const node = session.memoryStore.getNodeById(details!.memoryNodeId as string);
			expect(node).toBeDefined();
			expect(node!.cues.length).toBeGreaterThan(0);
		});
	});

	describe("Issue G: Deduplication on retry", () => {
		it("returns existing node when archiving the same firstKeptEntryId", () => {
			const memoryStore = MemoryStore.inMemory();
			const node1 = memoryStore.archiveSessionSegment({
				sessionId: "sess-retry",
				firstKeptEntryId: "entry-123",
				cues: ["cue1"],
				summary: "First attempt summary",
				messages: [{ role: "user", content: "msg", timestamp: 100 }],
			});

			// Retry archiving same span
			const node2 = memoryStore.archiveSessionSegment({
				sessionId: "sess-retry",
				firstKeptEntryId: "entry-123",
				cues: ["cue1"],
				summary: "Second attempt summary",
				messages: [{ role: "user", content: "msg", timestamp: 100 }],
			});

			expect(node2.id).toBe(node1.id);
			expect(memoryStore.size()).toBe(1);
		});
	});

	describe("Issue H: Multi-session file deletion and lazy loading", () => {
		it("deletes via tombstone without overwriting concurrent file appends", () => {
			const filePath = join(tempDir, "shared-store.jsonl");

			// Session A writes node 1
			const storeA = MemoryStore.open({ storageFilePath: filePath });
			storeA.archiveSessionSegment({
				id: "node-A1",
				sessionId: "session-A",
				cues: ["a1"],
				summary: "Node A1",
				messages: [{ role: "user", content: "A1", timestamp: 1 }],
			});

			// Session B writes node 2 directly to file
			const storeB = MemoryStore.open({ storageFilePath: filePath });
			storeB.archiveSessionSegment({
				id: "node-B1",
				sessionId: "session-B",
				cues: ["b1"],
				summary: "Node B1",
				messages: [{ role: "user", content: "B1", timestamp: 2 }],
			});

			// Session A deletes node A1
			storeA.deleteNode("node-A1");

			// Session C opens store: node-A1 is deleted, node-B1 is intact!
			const storeC = MemoryStore.open({ storageFilePath: filePath });
			expect(storeC.getNodeById("node-A1")).toBeUndefined();
			expect(storeC.getNodeById("node-B1")).toBeDefined();
		});

		it("skips torn or malformed JSONL entries gracefully", () => {
			const filePath = join(tempDir, "torn-store.jsonl");
			writeFileSync(filePath, '{"id":"valid-1","sessionId":"s1","cues":["c1"],"messages":[]}\n{"torn\n', "utf-8");

			const store = MemoryStore.open({ storageFilePath: filePath });
			expect(store.getNodeById("valid-1")).toBeDefined();
			expect(store.size()).toBe(1);
		});

		it("clears via tombstone without truncating shared file", () => {
			const filePath = join(tempDir, "shared-clear-store.jsonl");

			// Session A writes initial nodes
			const storeA = MemoryStore.open({ storageFilePath: filePath });
			storeA.archiveSessionSegment({
				id: "node-1",
				sessionId: "session-1",
				cues: ["c1"],
				summary: "Node 1",
				messages: [{ role: "user", content: "m1", timestamp: 1 }],
			});

			// Session A clears
			storeA.clear();
			expect(storeA.size()).toBe(0);

			// Session B appends after clear
			const storeB = MemoryStore.open({ storageFilePath: filePath });
			expect(storeB.size()).toBe(0);

			storeB.archiveSessionSegment({
				id: "node-2",
				sessionId: "session-2",
				cues: ["c2"],
				summary: "Node 2",
				messages: [{ role: "user", content: "m2", timestamp: 2 }],
			});

			// Session C loads from file: sees only node-2
			const storeC = MemoryStore.open({ storageFilePath: filePath });
			expect(storeC.size()).toBe(1);
			expect(storeC.getNodeById("node-1")).toBeUndefined();
			expect(storeC.getNodeById("node-2")).toBeDefined();
		});
	});

	describe("Issue B: memory_search bounding on large nodes and deep history chains", () => {
		it("bounds historyChain depth and total character output across multi-node chains", async () => {
			const memoryStore = MemoryStore.inMemory();
			const longBody = "x".repeat(3500);

			// Create 10 chained nodes, each with 30 messages
			let prevId: string | undefined;
			for (let i = 1; i <= 10; i++) {
				const nodeId = `chain-node-${i}`;
				const messages = Array.from({ length: 30 }, (_, m) => ({
					role: "user" as const,
					content: `Turn ${m + 1} in node ${i}: ${longBody}`,
					timestamp: 1000 + i * 100 + m,
				}));
				memoryStore.archiveSessionSegment({
					id: nodeId,
					sessionId: "deep-session",
					previousNodeId: prevId,
					cues: [`step-${i}`],
					summary: `Summary of node ${i}`,
					messages,
				});
				prevId = nodeId;
			}

			const tool = createMemorySearchTool(memoryStore);

			// 1. By default, historyChain defaults to maxDepth 5 and includeMessages: false
			const defaultResult = await tool.execute("call-chain-def", {
				nodeId: "chain-node-10",
				historyChain: true,
			});
			const defText = defaultResult.content[0].type === "text" ? defaultResult.content[0].text : "";
			expect(defaultResult.details?.nodeCount).toBe(5);
			expect(defText.length).toBeLessThan(10_000);

			// 2. Even when includeMessages: true and limit: 10 are requested, output is strictly bounded
			const fullResult = await tool.execute("call-chain-full", {
				nodeId: "chain-node-10",
				historyChain: true,
				limit: 10,
				includeMessages: true,
			});
			const fullText = fullResult.content[0].type === "text" ? fullResult.content[0].text : "";
			expect(fullText.length).toBeLessThanOrEqual(24_000);
			expect(fullText).toContain("[Output truncated");
		});

		it("bounds single-node direct inspection when node contains many large messages", async () => {
			const memoryStore = MemoryStore.inMemory();
			const longBody = "y".repeat(3500);

			const messages = Array.from({ length: 30 }, (_, m) => ({
				role: "user" as const,
				content: `Turn ${m + 1}: ${longBody}`,
				timestamp: 1000 + m,
			}));
			memoryStore.archiveSessionSegment({
				id: "large-node",
				sessionId: "large-sess",
				cues: ["heavy"],
				summary: "Node with 30 heavy turns",
				messages,
			});

			const tool = createMemorySearchTool(memoryStore);
			const result = await tool.execute("call-node-direct", {
				nodeId: "large-node",
			});
			const text = result.content[0].type === "text" ? result.content[0].text : "";

			// Output must be strictly bounded under 24,000 characters
			expect(text.length).toBeLessThanOrEqual(24_000);
			expect(text).toContain("earlier turn(s) omitted");
		});
	});
});
