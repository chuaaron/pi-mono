// Memory search tool for querying dark-layer archived session segments and tracing history chains.

import type { AgentMessage, AgentToolResult } from "@earendil-works/pi-agent-core";
import { type Static, Type } from "typebox";
import type { ExtensionContext, ToolDefinition } from "../extensions/types.ts";
import type { MemoryNode, MemorySearchResult, MemoryStore, SparseMatchMode } from "../memory/index.ts";
import { type TypeBoxAgentTool, wrapToolDefinition } from "./tool-definition-wrapper.ts";

const memorySearchSchema = Type.Object({
	query: Type.Optional(
		Type.String({
			description: "Search keywords or retrieval cues to search across archived sessions",
		}),
	),
	nodeId: Type.Optional(
		Type.String({
			description: "Direct memory node ID from an Archived Session Pointer",
		}),
	),
	sessionId: Type.Optional(
		Type.String({
			description: "Filter search to a specific session ID",
		}),
	),
	historyChain: Type.Optional(
		Type.Boolean({
			description: "When nodeId is specified, whether to trace and return the full preceding history chain",
		}),
	),
	mode: Type.Optional(
		Type.Union([Type.Literal("exact"), Type.Literal("all-tokens"), Type.Literal("any-token")], {
			description: "Matching mode for cue search ('exact', 'all-tokens', or 'any-token'). Defaults to 'any-token'",
		}),
	),
	limit: Type.Optional(
		Type.Integer({
			description: "Maximum number of search results to return (defaults to 5)",
		}),
	),
	includeMessages: Type.Optional(
		Type.Boolean({
			description:
				"Whether to return full turn messages. Defaults to true for single nodeId lookup, false for query search and history chains.",
		}),
	),
});

export type MemorySearchToolInput = Static<typeof memorySearchSchema>;

export const DEFAULT_MAX_SEARCH_TOTAL_CHARS = 24_000;

export const memorySearchToolSystemPromptContribution = {
	snippet: "Search archived conversation memory and history chains for past decisions, code changes, and context",
	guidelines: [
		"When the conversation context references an '[Archived Session Pointer: Node <id>]' or when you need historical evidence, decisions, or code changes from earlier compacted sessions, use the `memory_search` tool. If memory_search returns no results, you can conclude with high confidence that the issue/decision was not previously recorded.",
	],
} as const;

export interface MemorySearchToolDetails {
	nodeCount?: number;
	matchCount?: number;
	mode?: string;
	nodeIds?: string[];
}

export interface MemorySearchToolOptions {
	/** Maximum chain depth to trace when historyChain is requested. Defaults to 5. */
	maxDepth?: number;
	/** Maximum total characters across all returned text. Defaults to 24000. */
	maxTotalChars?: number;
	/** Maximum message turns displayed per node. Defaults to 10. */
	maxMessagesPerNode?: number;
}

function enforceTotalCharLimit(text: string, maxTotalChars = DEFAULT_MAX_SEARCH_TOTAL_CHARS): string {
	if (text.length <= maxTotalChars) return text;
	const excess = text.length - maxTotalChars;
	const half = Math.floor(Math.max(0, maxTotalChars - 200) / 2);
	const head = text.slice(0, half);
	const tail = text.slice(-half);
	return `${head}\n\n... [Output truncated (${excess} characters omitted). Re-query with specific nodeId, limit, or narrower query to inspect details] ...\n\n${tail}`;
}

function truncateHeadTail(text: string, maxChars = 2000): string {
	if (text.length <= maxChars) return text;
	const half = Math.floor(maxChars / 2);
	const head = text.slice(0, half);
	const tail = text.slice(-half);
	const excess = text.length - maxChars;
	return `${head}\n... [truncated ${excess} chars] ...\n${tail}`;
}

function formatMessage(msg: AgentMessage, maxChars = 2000): string {
	const messageRecord = msg as Record<string, unknown>;
	const role = typeof messageRecord.role === "string" ? messageRecord.role : "unknown";

	if (role === "user") {
		const content = messageRecord.content;
		if (typeof content === "string") {
			return `[User]\n${truncateHeadTail(content, maxChars)}`;
		}
		if (Array.isArray(content)) {
			const parts = content.map((part) => {
				if (typeof part === "object" && part !== null && "type" in part) {
					const p = part as { type: string; text?: string };
					if (p.type === "text" && typeof p.text === "string") return truncateHeadTail(p.text, maxChars);
					return `[${p.type}]`;
				}
				return truncateHeadTail(String(part), maxChars);
			});
			return `[User]\n${parts.join("\n")}`;
		}
		return `[User]\n${truncateHeadTail(JSON.stringify(content), maxChars)}`;
	}

	if (role === "assistant") {
		const content = messageRecord.content;
		const parts: string[] = [];
		if (Array.isArray(content)) {
			for (const part of content) {
				if (typeof part === "object" && part !== null && "type" in part) {
					const p = part as { type: string; text?: string; name?: string; arguments?: unknown; thinking?: string };
					if (p.type === "text" && typeof p.text === "string") {
						parts.push(truncateHeadTail(p.text, maxChars));
					} else if (p.type === "toolCall" && typeof p.name === "string") {
						const args = p.arguments ? truncateHeadTail(JSON.stringify(p.arguments), maxChars) : "{}";
						parts.push(`[Tool Call: ${p.name}(${args})]`);
					} else if (p.type === "thinking" && typeof p.thinking === "string") {
						parts.push(`[Thinking: ${truncateHeadTail(p.thinking, maxChars)}]`);
					}
				}
			}
		} else if (typeof content === "string") {
			parts.push(truncateHeadTail(content, maxChars));
		}
		return `[Assistant]\n${parts.join("\n")}`;
	}

	if (role === "toolResult") {
		const toolName =
			typeof messageRecord.toolName === "string"
				? messageRecord.toolName
				: typeof messageRecord.toolCallId === "string"
					? messageRecord.toolCallId
					: "tool";
		const isError = Boolean(messageRecord.isError);
		const content = messageRecord.content;
		let body = "";
		if (typeof content === "string") {
			body = truncateHeadTail(content, maxChars);
		} else if (Array.isArray(content)) {
			body = content
				.map((part) => {
					if (typeof part === "object" && part !== null && "text" in part) {
						return truncateHeadTail((part as { text: string }).text, maxChars);
					}
					return truncateHeadTail(JSON.stringify(part), maxChars);
				})
				.join("\n");
		} else if (content !== undefined) {
			body = truncateHeadTail(JSON.stringify(content), maxChars);
		}
		return `[Tool Result: ${toolName}${isError ? " (Error)" : ""}]\n${body}`;
	}

	return `[${role}]\n${truncateHeadTail(JSON.stringify(msg), maxChars)}`;
}

function formatMemoryNode(
	node: MemoryNode,
	options: { includeMessages?: boolean; maxMessageChars?: number; maxMessages?: number } = {},
): string {
	const { includeMessages = true, maxMessageChars = 2000, maxMessages = 10 } = options;
	const lines: string[] = [];
	lines.push(`Node ID: ${node.id}`);
	lines.push(`Session ID: ${node.sessionId}`);
	if (node.previousNodeId) {
		lines.push(`Previous Node: ${node.previousNodeId}`);
	}
	if (node.parentSessionId) {
		lines.push(`Parent Session: ${node.parentSessionId}`);
	}
	lines.push(`Timestamp: ${new Date(node.timestamp).toISOString()}`);
	lines.push(`Retrieval Cues: [${node.cues.join(", ")}]`);
	if (node.summary) {
		lines.push(`Summary: ${node.summary}`);
	}
	if (includeMessages) {
		if (node.messages.length > 0) {
			const displayMessages = node.messages.length > maxMessages ? node.messages.slice(-maxMessages) : node.messages;
			const omitted = node.messages.length - displayMessages.length;
			lines.push(`Messages (${node.messages.length} turn(s)):`);
			if (omitted > 0) {
				lines.push(`[... ${omitted} earlier turn(s) omitted; showing last ${maxMessages} turn(s) ...]`);
			}
			for (const msg of displayMessages) {
				lines.push("---");
				lines.push(formatMessage(msg, maxMessageChars));
			}
		} else {
			lines.push("Messages: (empty)");
		}
	} else {
		lines.push(
			`Messages: ${node.messages.length} turn(s) archived. (Re-query with nodeId: "${node.id}" for full message details)`,
		);
	}
	return lines.join("\n");
}

export function createMemorySearchToolDefinition(
	memoryStore: MemoryStore,
	options?: MemorySearchToolOptions,
): ToolDefinition<typeof memorySearchSchema, MemorySearchToolDetails | undefined> {
	return {
		name: "memory_search",
		label: "memory_search",
		description:
			"Search archived conversation memory and history chains for past decisions, code changes, and context. Allows direct pointer lookups by nodeId or sparse cue search by query.",
		promptSnippet: memorySearchToolSystemPromptContribution.snippet,
		promptGuidelines: [...memorySearchToolSystemPromptContribution.guidelines],
		parameters: memorySearchSchema,
		async execute(
			_toolCallId: string,
			params: MemorySearchToolInput,
			signal?: AbortSignal,
			_onUpdate?,
			_ctx?: ExtensionContext,
		): Promise<AgentToolResult<MemorySearchToolDetails | undefined>> {
			if (signal?.aborted) {
				throw new Error("Operation aborted");
			}

			const { nodeId, query, sessionId, historyChain, mode, limit, includeMessages } = params;

			// Path 1: Direct node ID lookup or history chain tracing
			if (nodeId) {
				if (historyChain) {
					const defaultMaxDepth = 5;
					const searchLimit = limit !== undefined && limit > 0 ? limit : (options?.maxDepth ?? defaultMaxDepth);
					const showMessages = includeMessages ?? false;
					const chain = memoryStore.getHistoryChain(nodeId, {
						direction: "root-to-leaf",
						maxDepth: searchLimit,
					});

					if (chain.length === 0) {
						return {
							content: [
								{
									type: "text",
									text: `No archived memory node found with ID "${nodeId}". This indicates that no prior record exists for this node identifier.`,
								},
							],
							details: { nodeCount: 0, nodeIds: [] },
						};
					}

					const parts = [
						`History chain (${chain.length} node(s)) leading to Node ${nodeId}:\n`,
						...chain.map(
							(node, index) =>
								`### Chain Hop ${index + 1}/${chain.length} (Node ${node.id})\n${formatMemoryNode(node, { includeMessages: showMessages, maxMessageChars: 2000, maxMessages: options?.maxMessagesPerNode ?? 10 })}`,
						),
					];

					return {
						content: [
							{
								type: "text",
								text: enforceTotalCharLimit(parts.join("\n\n"), options?.maxTotalChars),
							},
						],
						details: {
							nodeCount: chain.length,
							nodeIds: chain.map((n) => n.id),
						},
					};
				}

				const showMessages = includeMessages ?? true;
				const node = memoryStore.getNodeById(nodeId);
				if (!node) {
					return {
						content: [
							{
								type: "text",
								text: `No archived memory node found with ID "${nodeId}". This indicates that no prior record exists for this node identifier.`,
							},
						],
						details: { nodeCount: 0, nodeIds: [] },
					};
				}

				return {
					content: [
						{
							type: "text",
							text: enforceTotalCharLimit(
								`Archived memory node ${nodeId}:\n\n${formatMemoryNode(node, { includeMessages: showMessages, maxMessageChars: 4000, maxMessages: options?.maxMessagesPerNode ?? 10 })}`,
								options?.maxTotalChars,
							),
						},
					],
					details: {
						nodeCount: 1,
						nodeIds: [node.id],
					},
				};
			}

			// Path 2: Sparse cue keyword search
			if (query) {
				const defaultLimit = 5;
				const searchLimit = limit !== undefined && limit > 0 ? limit : defaultLimit;
				const showMessages = includeMessages ?? false;
				const matchMode: SparseMatchMode = mode ?? "any-token";
				const results: MemorySearchResult[] = memoryStore.searchByCue(query, {
					mode: matchMode,
					sessionId,
					limit: searchLimit,
				});

				if (results.length === 0) {
					const scope = sessionId ? ` (sessionId: "${sessionId}")` : "";
					return {
						content: [
							{
								type: "text",
								text: `No archived memory found matching query "${query}"${scope}. This indicates that no prior record exists for this retrieval cue.`,
							},
						],
						details: {
							matchCount: 0,
							mode: matchMode,
							nodeIds: [],
						},
					};
				}

				const parts = [
					`Found ${results.length} archived memory segment(s) matching "${query}":\n`,
					...results.map(
						(result, index) =>
							`### Result ${index + 1}/${results.length} (Score: ${result.score}, Matched Cues: [${result.matchedCues.join(", ")}])\n${formatMemoryNode(result.node, { includeMessages: showMessages, maxMessageChars: 2000, maxMessages: options?.maxMessagesPerNode ?? 10 })}`,
					),
				];

				return {
					content: [
						{
							type: "text",
							text: enforceTotalCharLimit(parts.join("\n\n"), options?.maxTotalChars),
						},
					],
					details: {
						matchCount: results.length,
						mode: matchMode,
						nodeIds: results.map((r) => r.node.id),
					},
				};
			}

			// Path 3: Session ID lookup without query or nodeId
			if (sessionId) {
				const defaultLimit = 5;
				const searchLimit = limit !== undefined && limit > 0 ? limit : defaultLimit;
				const showMessages = includeMessages ?? false;
				const allNodes = memoryStore.getNodesBySessionId(sessionId);
				const nodes = allNodes.slice(0, searchLimit);
				if (nodes.length === 0) {
					return {
						content: [
							{
								type: "text",
								text: `No archived memory found for session ID "${sessionId}". This indicates that no prior record exists for this session identifier.`,
							},
						],
						details: { nodeCount: 0, nodeIds: [] },
					};
				}

				const parts = [
					`Found ${nodes.length} archived memory segment(s) for session "${sessionId}":\n`,
					...nodes.map(
						(node, index) =>
							`### Segment ${index + 1}/${nodes.length} (Node ${node.id})\n${formatMemoryNode(node, { includeMessages: showMessages, maxMessageChars: 2000, maxMessages: options?.maxMessagesPerNode ?? 10 })}`,
					),
				];

				return {
					content: [
						{
							type: "text",
							text: enforceTotalCharLimit(parts.join("\n\n"), options?.maxTotalChars),
						},
					],
					details: {
						nodeCount: nodes.length,
						nodeIds: nodes.map((n) => n.id),
					},
				};
			}

			return {
				content: [
					{
						type: "text",
						text: "No search query or node ID provided. Specify a 'query' to search cues or a 'nodeId' to inspect an archived session segment.",
					},
				],
				details: { matchCount: 0 },
			};
		},
	};
}

export function createMemorySearchTool(
	memoryStore: MemoryStore,
	options?: MemorySearchToolOptions,
): TypeBoxAgentTool<typeof memorySearchSchema> {
	return wrapToolDefinition(createMemorySearchToolDefinition(memoryStore, options));
}
