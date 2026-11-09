import type { StreamFn } from "@earendil-works/pi-agent-core";
import { Agent } from "@earendil-works/pi-agent-core";
import { createAssistantMessageEventStream, fauxAssistantMessage, type Model } from "@earendil-works/pi-ai";
import { streamSimple } from "@earendil-works/pi-ai/compat";
import { describe, expect, it } from "vitest";
import { AgentSession } from "../src/core/agent-session.ts";
import { AuthStorage } from "../src/core/auth-storage.ts";
import {
	appendArchivedSessionPointer,
	type CompactionSettings,
	compact,
	formatArchivedSessionPointer,
	getMessagesToArchive,
	prepareCompaction,
	shouldCompact,
} from "../src/core/compaction/index.ts";
import { MemoryStore } from "../src/core/memory/index.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
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

function createMockStreamFn(summaryText = "Summary of archived segment."): StreamFn {
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

/** Mock stream that returns one canned text per call, repeating the last one. */
function createSequencedStreamFn(texts: string[]): StreamFn {
	let call = 0;
	return (model, context, options) => {
		const text = texts[Math.min(call, texts.length - 1)] ?? "";
		call += 1;
		return createMockStreamFn(text)(model, context, options);
	};
}

function createSessionWithMessages(messageCount = 6): SessionManager {
	const sessionManager = SessionManager.inMemory();
	for (let i = 0; i < messageCount; i++) {
		if (i % 2 === 0) {
			sessionManager.appendMessage({
				role: "user",
				content: `User query ${i}: implement authentication and database schema`,
				timestamp: Date.now() + i * 1000,
			});
		} else {
			sessionManager.appendMessage({
				...fauxAssistantMessage(`Assistant reply ${i}: created auth and user tables`),
				api: mockModel.api,
				provider: mockModel.provider,
				model: mockModel.id,
				usage: {
					input: 2000,
					output: 500,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 2500,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "stop",
				timestamp: Date.now() + i * 1000,
			});
		}
	}
	return sessionManager;
}

describe("watermark interception (capacityRatioThreshold)", () => {
	it("triggers compaction when context tokens exceed capacity ratio threshold", () => {
		const settings: CompactionSettings = {
			enabled: true,
			strategy: "sliding-memory",
			reserveTokens: 16384,
			keepRecentTokens: 20000,
			capacityRatioThreshold: 0.5,
		};

		// 200,000 * 0.5 = 100,000
		expect(shouldCompact(100_000, 200_000, settings)).toBe(true);
		expect(shouldCompact(150_000, 200_000, settings)).toBe(true);
		expect(shouldCompact(99_999, 200_000, settings)).toBe(false);
	});

	it("falls back to reserveTokens when capacity ratio is not met or not set", () => {
		const withoutRatio: CompactionSettings = {
			enabled: true,
			reserveTokens: 16384,
			keepRecentTokens: 20000,
		};

		// 200,000 - 16,384 = 183,616
		expect(shouldCompact(180_000, 200_000, withoutRatio)).toBe(false);
		expect(shouldCompact(183_617, 200_000, withoutRatio)).toBe(true);

		const withRatio: CompactionSettings = {
			enabled: true,
			strategy: "sliding-memory",
			reserveTokens: 10000,
			keepRecentTokens: 20000,
			capacityRatioThreshold: 0.8,
		};
		// reserveTokens trigger: > 190,000; capacity trigger: >= 160,000
		expect(shouldCompact(159_999, 200_000, withRatio)).toBe(false);
		expect(shouldCompact(160_000, 200_000, withRatio)).toBe(true);
	});

	it("returns false when compaction is disabled", () => {
		const disabledSettings: CompactionSettings = {
			enabled: false,
			reserveTokens: 16384,
			keepRecentTokens: 20000,
			capacityRatioThreshold: 0.5,
		};
		expect(shouldCompact(199_000, 200_000, disabledSettings)).toBe(false);
	});

	it("manages capacityRatioThreshold via SettingsManager", () => {
		const manager = SettingsManager.inMemory({
			compaction: {
				reserveTokens: 8192,
				keepRecentTokens: 10000,
				capacityRatioThreshold: 0.6,
				modelOverrides: {
					"test-provider/test-model": { capacityRatioThreshold: 0.4 },
				},
			},
		});

		expect(manager.getCompactionSettings()).toEqual({
			enabled: true,
			reserveTokens: 8192,
			keepRecentTokens: 10000,
			capacityRatioThreshold: 0.6,
		});

		expect(manager.getCompactionSettings(mockModel)).toEqual({
			enabled: true,
			reserveTokens: 8192,
			keepRecentTokens: 10000,
			capacityRatioThreshold: 0.4,
		});

		manager.setCompactionCapacityRatioThreshold(0.75);
		expect(manager.getCompactionCapacityRatioThreshold()).toBe(0.75);

		expect(() => manager.setCompactionCapacityRatioThreshold(-0.1)).toThrow();
		expect(() => manager.setCompactionCapacityRatioThreshold(1.5)).toThrow();
	});

	it("respects compaction strategy switch: native ignores capacityRatioThreshold", () => {
		const nativeSettings: CompactionSettings = {
			enabled: true,
			strategy: "native",
			reserveTokens: 16384,
			keepRecentTokens: 20000,
			capacityRatioThreshold: 0.5,
		};
		// Under native strategy, 50% capacity (100,000) does not trigger; only > 200,000 - 16,384 = 183,616 triggers.
		expect(shouldCompact(120_000, 200_000, nativeSettings)).toBe(false);
		expect(shouldCompact(183_617, 200_000, nativeSettings)).toBe(true);
	});

	it("manages strategy and memoryStoragePath via SettingsManager", () => {
		const defaultManager = SettingsManager.inMemory();
		expect(defaultManager.getCompactionStrategy()).toBe("native");
		expect(defaultManager.getCompactionSettings().strategy).toBeUndefined();

		const manager = SettingsManager.inMemory({
			compaction: {
				strategy: "native",
				memoryStoragePath: "/tmp/custom-memory.jsonl",
				modelOverrides: {
					"test-provider/test-model": { strategy: "sliding-memory" },
				},
			},
		});

		expect(manager.getCompactionStrategy()).toBe("native");
		expect(manager.getCompactionStrategy(mockModel)).toBe("sliding-memory");
		expect(manager.getCompactionMemoryStoragePath()).toBe("/tmp/custom-memory.jsonl");

		manager.setCompactionStrategy("sliding-memory");
		expect(manager.getCompactionStrategy()).toBe("sliding-memory");

		manager.setCompactionMemoryStoragePath(undefined);
		expect(manager.getCompactionMemoryStoragePath()).toBeUndefined();
	});
});

describe("lossless archival to MemoryStore & pointer injection", () => {
	it("archives messages losslessly to MemoryStore and injects pointer into summary", async () => {
		const sessionManager = createSessionWithMessages(6);
		const memoryStore = MemoryStore.inMemory();
		const entries = sessionManager.getBranch();

		const settings: CompactionSettings = {
			enabled: true,
			reserveTokens: 16384,
			keepRecentTokens: 1, // Cut early so messagesToSummarize has items
			capacityRatioThreshold: 0.5,
		};

		const preparation = prepareCompaction(entries, settings);
		expect(preparation).toBeDefined();
		expect(preparation!.messagesToSummarize.length).toBeGreaterThan(0);

		const result = await compact(
			preparation!,
			mockModel,
			"test-api-key",
			undefined,
			undefined,
			undefined,
			undefined,
			createMockStreamFn("Base summary text\n\n## Retrieval Cues\n- authentication\n- database-schema"),
			undefined,
			undefined,
			undefined,
			sessionManager.getSessionId(),
			memoryStore,
		);

		// 1. Verify details contains memoryNodeId
		const details = result.details;
		expect(details?.memoryNodeId).toBeDefined();
		const nodeId = details?.memoryNodeId ?? "";

		// 2. Verify MemoryStore holds the exact uncompressed messages
		const storedNode = memoryStore.getNodeById(nodeId);
		expect(storedNode).toBeDefined();
		expect(storedNode!.id).toBe(nodeId);
		expect(storedNode!.sessionId).toBe(sessionManager.getSessionId());
		expect(storedNode!.messages).toEqual(getMessagesToArchive(preparation!));
		expect(storedNode!.cues.length).toBeGreaterThan(0);

		// 3. Verify pointer is injected at the end of summary
		const expectedPointer = formatArchivedSessionPointer(storedNode!);
		expect(result.summary).toContain(expectedPointer);
		expect(result.summary.endsWith(expectedPointer)).toBe(true);

		// 4. Verify sparse search via cues finds the archived node
		const searchResults = memoryStore.searchByCue("authentication");
		expect(searchResults.length).toBeGreaterThan(0);
		expect(searchResults[0].node.id).toBe(nodeId);
		expect(searchResults[0].messages).toEqual(getMessagesToArchive(preparation!));
	});

	it("generates retrieval cues in a single turn when the summary omits the cues section", async () => {
		const sessionManager = createSessionWithMessages(6);
		const memoryStore = MemoryStore.inMemory();
		const preparation = prepareCompaction(sessionManager.getBranch(), {
			enabled: true,
			reserveTokens: 16384,
			keepRecentTokens: 1,
			capacityRatioThreshold: 0.5,
		})!;

		// First call: summary without the `## Retrieval Cues` section. Second call: cue fallback.
		const result = await compact(
			preparation,
			mockModel,
			"test-key",
			undefined,
			undefined,
			undefined,
			undefined,
			createSequencedStreamFn(["Summary that forgot the cues section.", "auth, db-schema, `migrations-v42`"]),
			undefined,
			undefined,
			undefined,
			sessionManager.getSessionId(),
			memoryStore,
		);

		const node = memoryStore.getNodeById(result.details!.memoryNodeId!)!;
		expect(node.cues).toEqual(["auth", "db-schema", "migrations-v42"]);
	});

	it("archives with no cues when the cue fallback also yields nothing", async () => {
		const sessionManager = createSessionWithMessages(6);
		const memoryStore = MemoryStore.inMemory();
		const preparation = prepareCompaction(sessionManager.getBranch(), {
			enabled: true,
			reserveTokens: 16384,
			keepRecentTokens: 1,
			capacityRatioThreshold: 0.5,
		})!;

		const result = await compact(
			preparation,
			mockModel,
			"test-key",
			undefined,
			undefined,
			undefined,
			undefined,
			createSequencedStreamFn(["Summary without cues.", ""]),
			undefined,
			undefined,
			undefined,
			sessionManager.getSessionId(),
			memoryStore,
		);

		const node = memoryStore.getNodeById(result.details!.memoryNodeId!)!;
		// No structural junk fallback: the node stays reachable only via its pointer.
		expect(node.cues).toEqual([]);
		expect(result.summary).toContain("Retrieval Cues: (none)");
	});

	it("formats pointer correctly according to specification", () => {
		const mockNode = {
			id: "node-12345",
			sessionId: "session-abcde",
			cues: ["auth", "database", "schema"],
		};

		const pointer = formatArchivedSessionPointer(mockNode);
		expect(pointer).toBe(
			"[Archived Session Pointer: Node node-12345 | Session session-abcde]\n" +
				"Retrieval Cues: auth, database, schema\n" +
				"Note: Full uncompressed details of this archived segment are preserved in the memory store. Use memory search to retrieve exact details.",
		);

		const withSummary = appendArchivedSessionPointer("Original summary", mockNode);
		expect(withSummary).toBe(`Original summary\n\n${pointer}`);
	});

	it("chains memory pointers across multiple compactions (history chain)", async () => {
		const sessionManager = createSessionWithMessages(4);
		const memoryStore = MemoryStore.inMemory();
		const settings: CompactionSettings = {
			enabled: true,
			reserveTokens: 16384,
			keepRecentTokens: 1,
			capacityRatioThreshold: 0.5,
		};

		// --- First Compaction ---
		const prep1 = prepareCompaction(sessionManager.getBranch(), settings)!;
		expect(prep1).toBeDefined();

		const result1 = await compact(
			prep1,
			mockModel,
			"test-key",
			undefined,
			undefined,
			undefined,
			undefined,
			createMockStreamFn("First summary"),
			undefined,
			undefined,
			undefined,
			sessionManager.getSessionId(),
			memoryStore,
		);

		sessionManager.appendCompaction(result1.summary, result1.firstKeptEntryId, result1.tokensBefore, result1.details);

		const node1Id = result1.details?.memoryNodeId ?? "";
		expect(node1Id).not.toBe("");

		// Add new conversational turns
		sessionManager.appendMessage({
			role: "user",
			content: "Next step: configure redis cache and token invalidation",
			timestamp: Date.now() + 10000,
		});
		sessionManager.appendMessage({
			...fauxAssistantMessage("Configured redis cluster and expiry keys"),
			api: mockModel.api,
			provider: mockModel.provider,
			model: mockModel.id,
			usage: {
				input: 2500,
				output: 600,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 3100,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: Date.now() + 11000,
		});

		// --- Second Compaction ---
		const prep2 = prepareCompaction(sessionManager.getBranch(), settings)!;
		expect(prep2).toBeDefined();

		const result2 = await compact(
			prep2,
			mockModel,
			"test-key",
			undefined,
			undefined,
			undefined,
			undefined,
			createMockStreamFn("Second summary"),
			undefined,
			undefined,
			undefined,
			sessionManager.getSessionId(),
			memoryStore,
		);

		sessionManager.appendCompaction(result2.summary, result2.firstKeptEntryId, result2.tokensBefore, result2.details);

		const node2Id = result2.details?.memoryNodeId ?? "";
		expect(node2Id).not.toBe("");
		expect(node2Id).not.toBe(node1Id);

		// Verify chain
		const node2 = memoryStore.getNodeById(node2Id)!;
		expect(node2.previousNodeId).toBe(node1Id);

		const chain = memoryStore.getHistoryChain(node2Id);
		expect(chain.length).toBe(2);
		expect(chain[0].id).toBe(node1Id);
		expect(chain[1].id).toBe(node2Id);
	});
});

describe("AgentSession integration with MemoryStore", () => {
	it("initializes with default or custom MemoryStore and exposes memoryStore getter", async () => {
		const customStore = MemoryStore.inMemory();
		const agent = new Agent({
			getApiKey: () => "test-key",
			streamFn: streamSimple,
			initialState: { model: mockModel },
		});
		const authStorage = AuthStorage.inMemory();
		const modelRuntime = getModelRuntime(await createInMemoryModelRegistry(authStorage));

		const sessionWithCustom = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(),
			settingsManager: SettingsManager.inMemory(),
			cwd: process.cwd(),
			modelRuntime,
			resourceLoader: createTestResourceLoader(),
			memoryStore: customStore,
		});

		expect(sessionWithCustom.memoryStore).toBe(customStore);

		const sessionWithDefault = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(),
			settingsManager: SettingsManager.inMemory(),
			cwd: process.cwd(),
			modelRuntime,
			resourceLoader: createTestResourceLoader(),
		});

		expect(sessionWithDefault.memoryStore).toBeInstanceOf(MemoryStore);
		expect(sessionWithDefault.memoryStore).not.toBe(customStore);
	});

	it("archives messages and injects pointer into CompactionEntry on session.compact()", async () => {
		const memoryStore = MemoryStore.inMemory();
		const sessionManager = createSessionWithMessages(6);
		const settingsManager = SettingsManager.inMemory({
			compaction: {
				enabled: true,
				strategy: "sliding-memory",
				reserveTokens: 16384,
				keepRecentTokens: 1, // Cut early so messages are summarized
				capacityRatioThreshold: 0.5,
			},
		});

		const agent = new Agent({
			getApiKey: () => "test-key",
			streamFn: createMockStreamFn("AgentSession compaction summary"),
			initialState: { model: mockModel },
		});
		const authStorage = AuthStorage.inMemory();
		const modelRuntime = getModelRuntime(await createInMemoryModelRegistry(authStorage));

		const session = new AgentSession({
			agent,
			sessionManager,
			settingsManager,
			cwd: process.cwd(),
			modelRuntime,
			resourceLoader: createTestResourceLoader(),
			memoryStore,
		});

		const result = await session.compact();
		expect(result).toBeDefined();
		expect(result.summary).toContain("[Archived Session Pointer:");

		// Check session entries
		const entries = sessionManager.getEntries();
		const compactionEntry = entries.find((e) => e.type === "compaction") as any;
		expect(compactionEntry).toBeDefined();
		expect(compactionEntry.details?.memoryNodeId).toBeDefined();

		const storedNode = memoryStore.getNodeById(compactionEntry.details.memoryNodeId);
		expect(storedNode).toBeDefined();
		expect(storedNode!.messages.length).toBeGreaterThan(0);
	});

	it("does not archive to MemoryStore when strategy is native", async () => {
		const memoryStore = MemoryStore.inMemory();
		const sessionManager = createSessionWithMessages(6);
		const settingsManager = SettingsManager.inMemory({
			compaction: {
				enabled: true,
				strategy: "native",
				reserveTokens: 16384,
				keepRecentTokens: 1,
			},
		});

		const agent = new Agent({
			getApiKey: () => "test-key",
			streamFn: createMockStreamFn("Native compaction summary"),
			initialState: { model: mockModel },
		});
		const authStorage = AuthStorage.inMemory();
		const modelRuntime = getModelRuntime(await createInMemoryModelRegistry(authStorage));

		const session = new AgentSession({
			agent,
			sessionManager,
			settingsManager,
			cwd: process.cwd(),
			modelRuntime,
			resourceLoader: createTestResourceLoader(),
			memoryStore,
		});

		const result = await session.compact();
		expect(result).toBeDefined();
		expect(result.summary).not.toContain("[Archived Session Pointer:");
		expect(memoryStore.size()).toBe(0);
	});
});
