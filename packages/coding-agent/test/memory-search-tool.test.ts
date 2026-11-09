import { mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent, type StreamFn } from "@earendil-works/pi-agent-core";
import { createAssistantMessageEventStream, fauxAssistantMessage, type Model } from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AgentSession } from "../src/core/agent-session.ts";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { MemoryStore } from "../src/core/memory/index.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import {
	createMemorySearchTool,
	createMemorySearchToolDefinition,
	memorySearchToolSystemPromptContribution,
} from "../src/core/tools/index.ts";
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

function createMockStreamFn(summaryText = "Archived conversation segment summary."): StreamFn {
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

describe("memory_search tool", () => {
	let memoryStore: MemoryStore;

	beforeEach(() => {
		memoryStore = MemoryStore.inMemory();
	});

	describe("tool definition and schema", () => {
		it("provides standard tool definition metadata and system prompt contribution", () => {
			const def = createMemorySearchToolDefinition(memoryStore);
			expect(def.name).toBe("memory_search");
			expect(def.label).toBe("memory_search");
			expect(def.promptSnippet).toBe(memorySearchToolSystemPromptContribution.snippet);
			expect(def.promptGuidelines).toEqual(
				expect.arrayContaining([...memorySearchToolSystemPromptContribution.guidelines]),
			);
			expect(def.parameters).toBeDefined();
		});

		it("wraps into TypeBoxAgentTool cleanly", () => {
			const tool = createMemorySearchTool(memoryStore);
			expect(tool.name).toBe("memory_search");
			expect(typeof tool.execute).toBe("function");
		});
	});

	describe("cue-based sparse keyword search", () => {
		beforeEach(() => {
			memoryStore.archiveSessionSegment({
				id: "node-auth",
				sessionId: "session-1",
				cues: ["jwt", "authentication", "bearer-token"],
				summary: "Implemented JWT authentication and bearer token verification.",
				messages: [
					{ role: "user", content: "How do we authenticate API requests?", timestamp: 1000 },
					{
						...fauxAssistantMessage("We use JWT with HS256 algorithm."),
						api: mockModel.api,
						provider: mockModel.provider,
						model: mockModel.id,
						content: [
							{ type: "text", text: "We use JWT with HS256 algorithm." },
							{ type: "toolCall", id: "call-1", name: "write", arguments: { path: "src/auth.ts" } },
						],
						timestamp: 1001,
					},
					{
						role: "toolResult",
						toolCallId: "call-1",
						toolName: "write",
						content: [{ type: "text", text: "File written successfully" }],
						isError: false,
						timestamp: 1002,
					},
				],
			});

			memoryStore.archiveSessionSegment({
				id: "node-db",
				sessionId: "session-1",
				cues: ["database", "postgres", "pool-size"],
				summary: "Configured PostgreSQL connection pool with max 20 connections.",
				messages: [
					{ role: "user", content: "Configure Postgres pool size", timestamp: 2000 },
					{
						...fauxAssistantMessage("Set pool max to 20 connections in pg pool config."),
						api: mockModel.api,
						provider: mockModel.provider,
						model: mockModel.id,
						timestamp: 2001,
					},
				],
			});
		});

		it("finds archived segments by keyword cue returning summary and cues by default", async () => {
			const tool = createMemorySearchTool(memoryStore);
			const result = await tool.execute("call-search-1", { query: "jwt authentication" });

			expect(result.content).toHaveLength(1);
			const text = result.content[0].type === "text" ? result.content[0].text : "";
			expect(text).toContain('Found 1 archived memory segment(s) matching "jwt authentication"');
			expect(text).toContain("Node ID: node-auth");
			expect(text).toContain("Implemented JWT authentication");
			expect(text).toContain(
				'Messages: 3 turn(s) archived. (Re-query with nodeId: "node-auth" for full message details)',
			);
			expect(result.details?.matchCount).toBe(1);
			expect(result.details?.nodeIds).toEqual(["node-auth"]);

			// When includeMessages is explicitly true, formatted turns are returned
			const fullResult = await tool.execute("call-search-full", {
				query: "jwt authentication",
				includeMessages: true,
			});
			const fullText = fullResult.content[0].type === "text" ? fullResult.content[0].text : "";
			expect(fullText).toContain("[User]\nHow do we authenticate API requests?");
			expect(fullText).toContain("[Assistant]\nWe use JWT with HS256 algorithm.");
			expect(fullText).toContain('[Tool Call: write({"path":"src/auth.ts"})]');
			expect(fullText).toContain("[Tool Result: write]\nFile written successfully");
		});

		it("supports search filtering by sessionId", async () => {
			memoryStore.archiveSessionSegment({
				id: "node-other-session",
				sessionId: "session-2",
				cues: ["jwt", "tokens"],
				summary: "Session 2 JWT notes",
				messages: [{ role: "user", content: "jwt in session 2", timestamp: 3000 }],
			});

			const tool = createMemorySearchTool(memoryStore);
			const resultSession1 = await tool.execute("call-search-2", { query: "jwt", sessionId: "session-1" });
			expect(resultSession1.details?.nodeIds).toEqual(["node-auth"]);

			const resultSession2 = await tool.execute("call-search-3", { query: "jwt", sessionId: "session-2" });
			expect(resultSession2.details?.nodeIds).toEqual(["node-other-session"]);
		});

		it("supports matching modes exact, all-tokens, and any-token", async () => {
			const tool = createMemorySearchTool(memoryStore);

			// exact match for full cue
			const exactHit = await tool.execute("call-exact", { query: "bearer-token", mode: "exact" });
			expect(exactHit.details?.matchCount).toBe(1);

			const exactMiss = await tool.execute("call-exact-miss", { query: "bearer", mode: "exact" });
			expect(exactMiss.details?.matchCount).toBe(0);

			// all-tokens requires all words to match
			const allTokensHit = await tool.execute("call-all", { query: "postgres database", mode: "all-tokens" });
			expect(allTokensHit.details?.matchCount).toBe(1);

			const allTokensMiss = await tool.execute("call-all-miss", { query: "postgres redis", mode: "all-tokens" });
			expect(allTokensMiss.details?.matchCount).toBe(0);
		});
	});

	describe("direct nodeId lookup & history chain tracing", () => {
		beforeEach(() => {
			memoryStore.archiveSessionSegment({
				id: "root-node",
				sessionId: "session-chain",
				cues: ["architecture", "init"],
				summary: "Initialized base architecture.",
				messages: [{ role: "user", content: "First requirement: use SQLite", timestamp: 100 }],
			});

			memoryStore.archiveSessionSegment({
				id: "child-node-1",
				sessionId: "session-chain",
				previousNodeId: "root-node",
				cues: ["schema", "tables"],
				summary: "Created users and sessions tables.",
				messages: [{ role: "user", content: "Second requirement: add users table", timestamp: 200 }],
			});

			memoryStore.archiveSessionSegment({
				id: "child-node-2",
				sessionId: "session-chain",
				previousNodeId: "child-node-1",
				cues: ["indices", "performance"],
				summary: "Added index on user_id.",
				messages: [{ role: "user", content: "Third requirement: index on user_id", timestamp: 300 }],
			});
		});

		it("retrieves a single node directly by nodeId", async () => {
			const tool = createMemorySearchTool(memoryStore);
			const result = await tool.execute("call-node-1", { nodeId: "child-node-1" });

			const text = result.content[0].type === "text" ? result.content[0].text : "";
			expect(text).toContain("Archived memory node child-node-1:");
			expect(text).toContain("Created users and sessions tables.");
			expect(text).toContain("Second requirement: add users table");
			expect(result.details?.nodeCount).toBe(1);
			expect(result.details?.nodeIds).toEqual(["child-node-1"]);
		});

		it("traces the full preceding history chain when historyChain is true", async () => {
			const tool = createMemorySearchTool(memoryStore);
			const result = await tool.execute("call-chain-1", { nodeId: "child-node-2", historyChain: true });

			const text = result.content[0].type === "text" ? result.content[0].text : "";
			expect(text).toContain("History chain (3 node(s)) leading to Node child-node-2:");
			expect(text).toContain("Chain Hop 1/3 (Node root-node)");
			expect(text).toContain("Chain Hop 2/3 (Node child-node-1)");
			expect(text).toContain("Chain Hop 3/3 (Node child-node-2)");
			expect(text).toContain("Initialized base architecture.");
			expect(text).toContain("Created users and sessions tables.");
			expect(text).toContain("Added index on user_id.");
			expect(result.details?.nodeCount).toBe(3);
			expect(result.details?.nodeIds).toEqual(["root-node", "child-node-1", "child-node-2"]);

			// When includeMessages: true is explicitly requested, turn messages are included
			const resultWithMsgs = await tool.execute("call-chain-2", {
				nodeId: "child-node-2",
				historyChain: true,
				includeMessages: true,
			});
			const textWithMsgs = resultWithMsgs.content[0].type === "text" ? resultWithMsgs.content[0].text : "";
			expect(textWithMsgs).toContain("First requirement: use SQLite");
			expect(textWithMsgs).toContain("Second requirement: add users table");
			expect(textWithMsgs).toContain("Third requirement: index on user_id");
		});
	});

	describe("identifiable retrieval failure", () => {
		it("returns explicit failure message when cue search produces no matches", async () => {
			const tool = createMemorySearchTool(memoryStore);
			const result = await tool.execute("call-fail-1", { query: "non_existent_cue_12345" });

			const text = result.content[0].type === "text" ? result.content[0].text : "";
			expect(text).toBe(
				'No archived memory found matching query "non_existent_cue_12345". This indicates that no prior record exists for this retrieval cue.',
			);
			expect(result.details?.matchCount).toBe(0);
		});

		it("returns explicit failure message when nodeId is not found", async () => {
			const tool = createMemorySearchTool(memoryStore);
			const result = await tool.execute("call-fail-2", { nodeId: "missing-node-id" });

			const text = result.content[0].type === "text" ? result.content[0].text : "";
			expect(text).toBe(
				'No archived memory node found with ID "missing-node-id". This indicates that no prior record exists for this node identifier.',
			);
			expect(result.details?.nodeCount).toBe(0);
		});

		it("returns explicit failure message when history chain starting nodeId is not found", async () => {
			const tool = createMemorySearchTool(memoryStore);
			const result = await tool.execute("call-fail-3", { nodeId: "missing-chain-id", historyChain: true });

			const text = result.content[0].type === "text" ? result.content[0].text : "";
			expect(text).toBe(
				'No archived memory node found with ID "missing-chain-id". This indicates that no prior record exists for this node identifier.',
			);
			expect(result.details?.nodeCount).toBe(0);
		});

		it("returns prompt guidance when neither query nor nodeId is supplied", async () => {
			const tool = createMemorySearchTool(memoryStore);
			const result = await tool.execute("call-fail-4", {});

			const text = result.content[0].type === "text" ? result.content[0].text : "";
			expect(text).toContain("No search query or node ID provided");
		});
	});
});

describe("AgentSession integration with memory_search", () => {
	let tempDir: string;
	let sessionManager: SessionManager;
	let settingsManager: SettingsManager;
	let session: AgentSession;

	beforeEach(async () => {
		tempDir = join(tmpdir(), `pi-test-memory-search-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		mkdirSync(tempDir, { recursive: true });

		sessionManager = SessionManager.inMemory(tempDir);
		settingsManager = SettingsManager.inMemory({
			compaction: {
				enabled: true,
				strategy: "sliding-memory",
				keepRecentTokens: 1,
			},
		});

		const authStorage = AuthStorage.inMemory();
		const modelRuntime = getModelRuntime(await createInMemoryModelRegistry(authStorage));

		const agent = new Agent({
			getApiKey: () => "test-key",
			streamFn: createMockStreamFn(
				"Summary of earlier decisions and code changes.\n\n## Retrieval Cues\n- database schema migration\n- version 42 enum types",
			),
			initialState: {
				model: mockModel,
				systemPrompt: "",
				tools: [],
			},
		});

		session = new AgentSession({
			agent,
			sessionManager,
			settingsManager,
			resourceLoader: createTestResourceLoader(),
			modelRuntime,
			cwd: tempDir,
		});
	});

	afterEach(() => {
		session.dispose();
		if (tempDir) {
			rmSync(tempDir, { recursive: true, force: true });
		}
	});

	it("registers memory_search bound to session.memoryStore in registered and callable tools", () => {
		const all = session.getAllTools().map((t) => t.name);
		expect(all).toContain("memory_search");

		const def = session.getToolDefinition("memory_search");
		expect(def).toBeDefined();
		expect(def?.name).toBe("memory_search");

		session.setActiveToolsByName([...session.getActiveToolNames(), "memory_search"]);
		expect(session.getCallableToolNames()).toContain("memory_search");
	});

	it("contributes retrieval guidelines to system prompt", () => {
		expect(session.systemPrompt).toContain("use the `memory_search` tool");
		expect(session.systemPrompt).toContain("[Archived Session Pointer: Node <id>]");
	});

	it("retrieves archived turn messages via memory_search after sliding window compaction", async () => {
		// 1. Simulate conversation turns in the session
		sessionManager.appendMessage({
			role: "user",
			content: "Decision: Migrate database schema to version 42 with strict enum types",
			timestamp: Date.now() - 5000,
		});
		sessionManager.appendMessage({
			...fauxAssistantMessage("Acknowledged decision to migrate schema to v42."),
			api: mockModel.api,
			provider: mockModel.provider,
			model: mockModel.id,
			timestamp: Date.now() - 4000,
		});
		sessionManager.appendMessage({
			role: "user",
			content: "Next step: create migrations for version 42",
			timestamp: Date.now() - 3000,
		});
		sessionManager.appendMessage({
			...fauxAssistantMessage("Creating migration files now."),
			api: mockModel.api,
			provider: mockModel.provider,
			model: mockModel.id,
			timestamp: Date.now() - 2000,
		});

		// 2. Trigger compaction to archive previous turns to MemoryStore
		const compactionResult = await session.compact();
		expect(compactionResult.summary).toBeDefined();

		// Verify pointer was generated
		const pointerMatch = compactionResult.summary.match(
			/\[Archived Session Pointer:\s*Node\s*([^\s|]+)\s*\|\s*Session\s*([^\]]+)]/,
		);
		expect(pointerMatch).not.toBeNull();
		const archivedNodeId = pointerMatch![1];

		// 3. Obtain memory_search tool from session
		const memorySearchTool =
			session.agent.state.tools.find((t) => t.name === "memory_search") ??
			createMemorySearchTool(session.memoryStore);

		// 4. Trace the archived turn by its pointer nodeId
		const nodeResult = await memorySearchTool.execute("test-call-1", {
			nodeId: archivedNodeId,
		});
		const nodeText = nodeResult.content[0].type === "text" ? nodeResult.content[0].text : "";
		expect(nodeText).toContain(`Archived memory node ${archivedNodeId}:`);
		expect(nodeText).toContain("Migrate database schema to version 42");
		expect(nodeText).toContain("Acknowledged decision to migrate schema to v42.");

		// 5. Query by keyword cue (returns summary + cues + re-query hint by default)
		const cueResult = await memorySearchTool.execute("test-call-2", {
			query: "database schema version 42",
		});
		const cueText = cueResult.content[0].type === "text" ? cueResult.content[0].text : "";
		expect(cueText).toContain(`Found 1 archived memory segment(s) matching "database schema version 42"`);
		expect(cueText).toContain("Summary of earlier decisions and code changes");
		expect(cueText).toContain(`Re-query with nodeId: "${archivedNodeId}" for full message details`);

		// 6. Query for non-existent topic returns identifiable failure
		const missResult = await memorySearchTool.execute("test-call-3", {
			query: "quantum teleportation protocol",
		});
		const missText = missResult.content[0].type === "text" ? missResult.content[0].text : "";
		expect(missText).toContain('No archived memory found matching query "quantum teleportation protocol"');
	});
});
