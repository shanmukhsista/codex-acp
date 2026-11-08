import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { ServerNotification } from '../../app-server';
import { createCodexMockTestFixture, createTestEventHandler, createTestSessionState, type CodexMockTestFixture } from '../acp-test-utils';
import { PromptTokenUsage, toPromptUsage, type TokenCount } from '../../TokenCount';
import type { TokenUsageBreakdown } from '../../app-server/v2';
import type { SessionState } from '../../CodexAcpServer';
import { ACPSessionConnection } from '../../ACPSessionConnection';
import type { Model } from '../../app-server/v2';
import { CodexSubagentEventRouter } from '../../subagents/CodexSubagentEventRouter';

function createTokenUsageNotification(
    sessionId: string,
    tokenUsage: {
        total: TokenUsageBreakdown;
        last: TokenUsageBreakdown;
        modelContextWindow: number | null;
    },
    turnId = 'turn-id',
): ServerNotification {
    return {
        method: 'thread/tokenUsage/updated',
        params: {
            threadId: sessionId,
            turnId,
            tokenUsage,
        },
    };
}

/** Model output that precedes the usage update of a model request. */
function reasoningStarted(threadId: string, id = 'reasoning-id'): ServerNotification {
    return {
        method: 'item/started',
        params: {
            threadId,
            turnId: 'turn-id',
            startedAtMs: 0,
            item: { type: 'reasoning', id, summary: [], content: [] },
        },
    };
}

function breakdown(totalTokens: number, inputTokens: number, cachedInputTokens: number, outputTokens: number): TokenUsageBreakdown {
    return {totalTokens, inputTokens, cachedInputTokens, cacheWriteInputTokens: 0, outputTokens, reasoningOutputTokens: 0};
}

describe('Token Usage Events', () => {
    let mockFixture: CodexMockTestFixture;
    const sessionId = 'test-session-id';

    beforeEach(() => {
        mockFixture = createCodexMockTestFixture();
        vi.clearAllMocks();
    });
    describe('PromptResponse usage', () => {
        function setupPromptWithTokenUsage(notifications: ServerNotification[], turnStatus: string = "completed") {
            const codexAcpAgent = mockFixture.getCodexAcpAgent();

            mockFixture.getCodexAppServerClient().turnStart = vi.fn().mockResolvedValue({
                turn: { id: "turn-id", items: [], status: "inProgress", error: null }
            });

            // awaitTurnCompleted sends notifications before resolving
            mockFixture.getCodexAppServerClient().awaitTurnCompleted = vi.fn().mockImplementation(async () => {
                // Send notifications during turn (after handler is registered)
                for (const notification of [reasoningStarted(sessionId), ...notifications]) {
                    mockFixture.sendServerNotification(notification);
                }
                return {
                    threadId: sessionId,
                    turn: { id: "turn-id", items: [], status: turnStatus, error: null }
                };
            });

            vi.spyOn(codexAcpAgent, 'getSessionState').mockReturnValue(createTestSessionState({ sessionId }));

            return codexAcpAgent;
        }

        it('should include token_count in PromptResponse on end_turn', async () => {
            const tokenUsageNotification = createTokenUsageNotification(sessionId, {
                total: {
                    totalTokens: 5000,
                    inputTokens: 4000,
                    cachedInputTokens: 1000,
                    cacheWriteInputTokens: 0,
                    outputTokens: 900,
                    reasoningOutputTokens: 100,
                },
                last: {
                    totalTokens: 2500,
                    inputTokens: 2000,
                    cachedInputTokens: 500,
                    cacheWriteInputTokens: 0,
                    outputTokens: 450,
                    reasoningOutputTokens: 50,
                },
                modelContextWindow: 128000,
            });

            const codexAcpAgent = setupPromptWithTokenUsage([tokenUsageNotification]);

            const response = await codexAcpAgent.prompt({
                sessionId,
                prompt: [{ type: 'text', text: 'test prompt' }],
            });

            await expect(`${JSON.stringify(response, null, 2)}\n`).toMatchFileSnapshot(
                'data/token-usage-end-turn.json'
            );
        });

        it('should include token_count in PromptResponse on cancelled', async () => {
            const tokenUsageNotification = createTokenUsageNotification(sessionId, {
                total: {
                    totalTokens: 3000,
                    inputTokens: 2500,
                    cachedInputTokens: 0,
                    cacheWriteInputTokens: 0,
                    outputTokens: 500,
                    reasoningOutputTokens: 0,
                },
                last: {
                    totalTokens: 1500,
                    inputTokens: 1200,
                    cachedInputTokens: 0,
                    cacheWriteInputTokens: 0,
                    outputTokens: 300,
                    reasoningOutputTokens: 0,
                },
                modelContextWindow: 128000,
            });

            const codexAcpAgent = setupPromptWithTokenUsage([tokenUsageNotification], "interrupted");

            const response = await codexAcpAgent.prompt({
                sessionId,
                prompt: [{ type: 'text', text: 'test prompt' }],
            });

            await expect(`${JSON.stringify(response, null, 2)}\n`).toMatchFileSnapshot(
                'data/token-usage-cancelled.json'
            );
        });

        it('should return null token_count when no token usage event received', async () => {
            const codexAcpAgent = setupPromptWithTokenUsage([]);

            const response = await codexAcpAgent.prompt({
                sessionId,
                prompt: [{ type: 'text', text: 'test prompt' }],
            });

            await expect(`${JSON.stringify(response, null, 2)}\n`).toMatchFileSnapshot(
                'data/token-usage-null.json'
            );
        });

        it('should report the whole turn from multiple updates', async () => {
            const notifications: ServerNotification[] = [
                createTokenUsageNotification(sessionId, {
                    total: { totalTokens: 1000, inputTokens: 800, cachedInputTokens: 0, cacheWriteInputTokens: 0, outputTokens: 200, reasoningOutputTokens: 0 },
                    last: { totalTokens: 1000, inputTokens: 800, cachedInputTokens: 0, cacheWriteInputTokens: 0, outputTokens: 200, reasoningOutputTokens: 0 },
                    modelContextWindow: 128000,
                }),
                createTokenUsageNotification(sessionId, {
                    total: { totalTokens: 2000, inputTokens: 1600, cachedInputTokens: 0, cacheWriteInputTokens: 0, outputTokens: 400, reasoningOutputTokens: 0 },
                    last: { totalTokens: 1000, inputTokens: 800, cachedInputTokens: 0, cacheWriteInputTokens: 0, outputTokens: 200, reasoningOutputTokens: 0 },
                    modelContextWindow: 128000,
                }),
                createTokenUsageNotification(sessionId, {
                    total: { totalTokens: 3500, inputTokens: 2800, cachedInputTokens: 500, cacheWriteInputTokens: 0, outputTokens: 600, reasoningOutputTokens: 100 },
                    last: { totalTokens: 1500, inputTokens: 1200, cachedInputTokens: 500, cacheWriteInputTokens: 0, outputTokens: 200, reasoningOutputTokens: 100 },
                    modelContextWindow: 128000,
                }),
            ];

            const codexAcpAgent = setupPromptWithTokenUsage(notifications);

            const response = await codexAcpAgent.prompt({
                sessionId,
                prompt: [{ type: 'text', text: 'test prompt' }],
            });

            await expect(`${JSON.stringify(response, null, 2)}\n`).toMatchFileSnapshot(
                'data/token-usage-multiple-updates.json'
            );
        });
    });

    describe('PromptResponse usage of the whole turn', () => {
        function setupPrompts(
            turns: ServerNotification[][],
            state: Partial<SessionState> = {},
            startWithModelOutput = true,
        ) {
            const codexAcpAgent = mockFixture.getCodexAcpAgent();
            let turn = 0;
            mockFixture.getCodexAppServerClient().turnStart = vi.fn().mockImplementation(async () => ({
                turn: { id: `turn-${turn + 1}`, items: [], status: "inProgress", error: null },
            }));
            mockFixture.getCodexAppServerClient().awaitTurnCompleted = vi.fn().mockImplementation(async () => {
                const notifications = turns[turn++] ?? [];
                for (const notification of startWithModelOutput ? [reasoningStarted(sessionId), ...notifications] : notifications) {
                    mockFixture.sendServerNotification(notification);
                }
                return {
                    threadId: sessionId,
                    turn: { id: `turn-${turn}`, items: [], status: "completed", error: null },
                };
            });
            vi.spyOn(codexAcpAgent, 'getSessionState').mockReturnValue(createTestSessionState({ sessionId, ...state }));
            return async () => await codexAcpAgent.prompt({
                sessionId,
                prompt: [{ type: 'text', text: 'test prompt' }],
            });
        }

        it('sums every model request of the turn', async () => {
            const prompt = setupPrompts([[
                createTokenUsageNotification(sessionId, {
                    total: breakdown(1000, 800, 0, 200),
                    last: breakdown(1000, 800, 0, 200),
                    modelContextWindow: 128000,
                }),
                createTokenUsageNotification(sessionId, {
                    total: breakdown(2300, 1900, 700, 400),
                    last: breakdown(1300, 1100, 700, 200),
                    modelContextWindow: 128000,
                }),
            ]]);

            const response = await prompt();

            expect(response.usage).toEqual({
                totalTokens: 2300,
                inputTokens: 1200,
                cachedReadTokens: 700,
                cachedWriteTokens: 0,
                outputTokens: 400,
                thoughtTokens: 0,
            });
        });

        it('reports only the usage of the turn after an earlier turn', async () => {
            const prompt = setupPrompts([
                [
                    createTokenUsageNotification(sessionId, {
                        total: breakdown(1000, 800, 0, 200),
                        last: breakdown(1000, 800, 0, 200),
                        modelContextWindow: 128000,
                    }),
                ],
                [
                    createTokenUsageNotification(sessionId, {
                        total: breakdown(2500, 2000, 600, 500),
                        last: breakdown(1500, 1200, 600, 300),
                        modelContextWindow: 128000,
                    }),
                    createTokenUsageNotification(sessionId, {
                        total: breakdown(4500, 3700, 1800, 800),
                        last: breakdown(2000, 1700, 1200, 300),
                        modelContextWindow: 128000,
                    }),
                ],
            ]);

            await prompt();
            const response = await prompt();

            expect(response.usage).toEqual({
                totalTokens: 3500,
                inputTokens: 1100,
                cachedReadTokens: 1800,
                cachedWriteTokens: 0,
                outputTokens: 600,
                thoughtTokens: 0,
            });
        });

        it('subtracts the total that the session had when the turn started', async () => {
            const prompt = setupPrompts([[
                createTokenUsageNotification(sessionId, {
                    // Codex sent no update for an earlier request of this turn, so `last` misses it.
                    total: breakdown(6000, 5000, 1000, 1000),
                    last: breakdown(1000, 800, 500, 200),
                    modelContextWindow: 128000,
                }),
            ]], {
                totalTokenUsage: {
                    totalTokens: 3000,
                    inputTokens: 2500,
                    cachedInputTokens: 0,
                    cacheWriteInputTokens: 0,
                    outputTokens: 500,
                    reasoningOutputTokens: 0,
                },
            });

            const response = await prompt();

            expect(response.usage?.totalTokens).toBe(3000);
            expect(response.usage?.inputTokens).toBe(1500);
            expect(response.usage?.cachedReadTokens).toBe(1000);
        });

        it('does not count the inherited history of a forked or resumed thread', async () => {
            // The first total of a forked or resumed thread already includes the usage of its history.
            const prompt = setupPrompts([[
                createTokenUsageNotification(sessionId, {
                    total: breakdown(102000, 100000, 80000, 2000),
                    last: breakdown(2000, 1800, 1500, 200),
                    modelContextWindow: 128000,
                }),
                createTokenUsageNotification(sessionId, {
                    total: breakdown(105000, 102700, 82000, 2300),
                    last: breakdown(3000, 2700, 2000, 300),
                    modelContextWindow: 128000,
                }),
            ]]);

            const response = await prompt();

            expect(response.usage?.totalTokens).toBe(5000);
            expect(response.usage?.inputTokens).toBe(1000);
            expect(response.usage?.cachedReadTokens).toBe(3500);
            expect(response.usage?.outputTokens).toBe(500);
        });

        it('does not count the inherited history that Codex sends again when the turn of a resumed thread starts', async () => {
            const prompt = setupPrompts([[
                // Sent again with the rate limits when the first request starts, before any model output.
                createTokenUsageNotification(sessionId, {
                    total: breakdown(100000, 98000, 80000, 2000),
                    last: breakdown(40000, 39000, 30000, 1000),
                    modelContextWindow: 128000,
                }),
                reasoningStarted(sessionId),
                createTokenUsageNotification(sessionId, {
                    total: breakdown(103000, 100700, 82000, 2300),
                    last: breakdown(3000, 2700, 2000, 300),
                    modelContextWindow: 128000,
                }),
            ]], {}, false);

            const response = await prompt();

            expect(response.usage?.totalTokens).toBe(3000);
            expect(response.usage?.inputTokens).toBe(700);
            expect(response.usage?.cachedReadTokens).toBe(2000);
        });

        it('does not count a history of one request that Codex sends again when the turn starts', async () => {
            const prompt = setupPrompts([[
                createTokenUsageNotification(sessionId, {
                    total: breakdown(1000, 800, 0, 200),
                    last: breakdown(1000, 800, 0, 200),
                    modelContextWindow: 128000,
                }),
                reasoningStarted(sessionId),
                createTokenUsageNotification(sessionId, {
                    total: breakdown(1300, 1050, 0, 250),
                    last: breakdown(300, 250, 0, 50),
                    modelContextWindow: 128000,
                }),
            ]], {}, false);

            const response = await prompt();

            expect(response.usage?.totalTokens).toBe(300);
        });

        it('counts the compaction of a resumed thread but not the history sent again before it', async () => {
            const prompt = setupPrompts([[
                {
                    method: 'item/started',
                    params: {
                        threadId: sessionId,
                        turnId: 'turn-id',
                        startedAtMs: 0,
                        item: { type: 'contextCompaction', id: 'compaction-id' },
                    },
                },
                createTokenUsageNotification(sessionId, {
                    total: breakdown(100000, 98000, 80000, 2000),
                    last: breakdown(40000, 39000, 30000, 1000),
                    modelContextWindow: 128000,
                }),
                createTokenUsageNotification(sessionId, {
                    total: breakdown(102000, 99500, 80000, 2500),
                    last: breakdown(2000, 1500, 0, 500),
                    modelContextWindow: 128000,
                }),
            ]], {}, false);

            const response = await prompt();

            expect(response.usage?.totalTokens).toBe(2000);
        });

        it('counts the first request of a new thread even before its output is reported', async () => {
            const prompt = setupPrompts([[
                createTokenUsageNotification(sessionId, {
                    total: breakdown(1000, 800, 0, 200),
                    last: breakdown(1000, 800, 0, 200),
                    modelContextWindow: 128000,
                }),
            ]], { threadHasHistory: false }, false);

            const response = await prompt();

            expect(response.usage?.totalTokens).toBe(1000);
        });

        it('does not count a re-sent update twice', async () => {
            const first = createTokenUsageNotification(sessionId, {
                total: breakdown(1000, 800, 0, 200),
                last: breakdown(1000, 800, 0, 200),
                modelContextWindow: 128000,
            });
            const second = createTokenUsageNotification(sessionId, {
                total: breakdown(2500, 2000, 0, 500),
                last: breakdown(1500, 1200, 0, 300),
                modelContextWindow: 128000,
            });
            const prompt = setupPrompts([[first, second, second]]);

            const response = await prompt();

            expect(response.usage?.totalTokens).toBe(2500);
        });

        it('does not report the usage of the previous prompt for a prompt without a turn', async () => {
            const prompt = setupPrompts([[
                createTokenUsageNotification(sessionId, {
                    total: breakdown(1000, 800, 0, 200),
                    last: breakdown(1000, 800, 0, 200),
                    modelContextWindow: 128000,
                }),
            ]]);
            await prompt();

            const response = await mockFixture.getCodexAcpAgent().prompt({
                sessionId,
                prompt: [{ type: 'text', text: '/status' }],
            });

            expect(response.usage).toBeNull();
        });

        describe('_meta.quota', () => {
            const quotaOf = (response: { _meta?: Record<string, unknown> | null }) =>
                (response._meta as { quota: { token_count: TokenCount | null, model_usage: Array<{ model: string, token_count: TokenCount }> } }).quota;

            it('reports the whole turn after an earlier turn, as PromptResponse.usage does', async () => {
                const prompt = setupPrompts([
                    [createTokenUsageNotification(sessionId, {
                        total: breakdown(1000, 800, 0, 200),
                        last: breakdown(1000, 800, 0, 200),
                        modelContextWindow: 128000,
                    })],
                    [
                        createTokenUsageNotification(sessionId, {
                            total: breakdown(2500, 2000, 600, 500),
                            last: breakdown(1500, 1200, 600, 300),
                            modelContextWindow: 128000,
                        }),
                        createTokenUsageNotification(sessionId, {
                            total: breakdown(4500, 3700, 1800, 800),
                            last: breakdown(2000, 1700, 1200, 300),
                            modelContextWindow: 128000,
                        }),
                    ],
                ]);

                await prompt();
                const response = await prompt();

                const quota = quotaOf(response);
                expect(quota.token_count).toMatchObject({totalTokens: 3500, inputTokens: 1100, cachedInputTokens: 1800, outputTokens: 600});
                expect(toPromptUsage(quota.token_count!)).toEqual(response.usage);
                expect(quota.model_usage).toEqual([{model: expect.any(String), token_count: quota.token_count}]);
            });

            it('leaves out the inherited history of a forked or resumed thread, as PromptResponse.usage does', async () => {
                const prompt = setupPrompts([[
                    createTokenUsageNotification(sessionId, {
                        total: breakdown(102000, 100000, 80000, 2000),
                        last: breakdown(2000, 1800, 1500, 200),
                        modelContextWindow: 128000,
                    }),
                    createTokenUsageNotification(sessionId, {
                        total: breakdown(105000, 102700, 82000, 2300),
                        last: breakdown(3000, 2700, 2000, 300),
                        modelContextWindow: 128000,
                    }),
                ]]);

                const response = await prompt();

                expect(quotaOf(response).token_count?.totalTokens).toBe(5000);
                expect(toPromptUsage(quotaOf(response).token_count!)).toEqual(response.usage);
            });

            it('reports no usage for a prompt without a turn', async () => {
                const prompt = setupPrompts([[
                    createTokenUsageNotification(sessionId, {
                        total: breakdown(1000, 800, 0, 200),
                        last: breakdown(1000, 800, 0, 200),
                        modelContextWindow: 128000,
                    }),
                ]]);
                await prompt();

                const response = await mockFixture.getCodexAcpAgent().prompt({
                    sessionId,
                    prompt: [{ type: 'text', text: '/status' }],
                });

                expect(quotaOf(response)).toEqual({token_count: null, model_usage: []});
            });
        });

        it('never reports negative usage when the thread total goes back', async () => {
            const prompt = setupPrompts([[
                createTokenUsageNotification(sessionId, {
                    total: breakdown(1500, 1200, 0, 300),
                    last: breakdown(1500, 1200, 0, 300),
                    modelContextWindow: 128000,
                }),
                createTokenUsageNotification(sessionId, {
                    total: breakdown(2500, 2000, 0, 500),
                    last: breakdown(1000, 800, 0, 200),
                    modelContextWindow: 128000,
                }),
            ]], {
                totalTokenUsage: {
                    totalTokens: 9000,
                    inputTokens: 8000,
                    cachedInputTokens: 0,
                    cacheWriteInputTokens: 0,
                    outputTokens: 1000,
                    reasoningOutputTokens: 0,
                },
            });

            const response = await prompt();

            // The total cannot be compared with the start, so the requests of the turn are summed.
            expect(response.usage?.totalTokens).toBe(2500);
            expect(response.usage?.inputTokens).toBe(2000);
        });
    });

    describe('session/update usage_update', () => {
        function setupPromptAndReturnEvents(notifications: ServerNotification[], turnStatus: string = "completed") {
            const codexAcpAgent = mockFixture.getCodexAcpAgent();

            mockFixture.getCodexAppServerClient().turnStart = vi.fn().mockResolvedValue({
                turn: { id: "turn-id", items: [], status: "inProgress", error: null }
            });

            mockFixture.getCodexAppServerClient().awaitTurnCompleted = vi.fn().mockImplementation(async () => {
                for (const notification of notifications) {
                    mockFixture.sendServerNotification(notification);
                }
                return {
                    threadId: sessionId,
                    turn: { id: "turn-id", items: [], status: turnStatus, error: null }
                };
            });

            vi.spyOn(codexAcpAgent, 'getSessionState').mockReturnValue(createTestSessionState({ sessionId }));

            return async () => {
                await codexAcpAgent.prompt({
                    sessionId,
                    prompt: [{ type: 'text', text: 'test prompt' }],
                });
                return mockFixture.getAcpConnectionEvents([]);
            };
        }

        it('should emit usage_update with latest turn usage as a context proxy', async () => {
            const events = await setupPromptAndReturnEvents([
                createTokenUsageNotification(sessionId, {
                    total: {
                        totalTokens: 5000,
                        inputTokens: 4000,
                        cachedInputTokens: 1000,
                        cacheWriteInputTokens: 0,
                        outputTokens: 900,
                        reasoningOutputTokens: 100,
                    },
                    last: {
                        totalTokens: 2500,
                        inputTokens: 2000,
                        cachedInputTokens: 500,
                        cacheWriteInputTokens: 0,
                        outputTokens: 450,
                        reasoningOutputTokens: 50,
                    },
                    modelContextWindow: 128000,
                }),
            ])();

            await expect(`${JSON.stringify(events[0], null, 2)}\n`).toMatchFileSnapshot('data/token-usage-session-update.json');
        });

        it('should emit latest turn usage from multiple updates', async () => {
            const events = await setupPromptAndReturnEvents([
                createTokenUsageNotification(sessionId, {
                    total: { totalTokens: 1000, inputTokens: 800, cachedInputTokens: 0, cacheWriteInputTokens: 0, outputTokens: 200, reasoningOutputTokens: 0 },
                    last: { totalTokens: 1000, inputTokens: 800, cachedInputTokens: 0, cacheWriteInputTokens: 0, outputTokens: 200, reasoningOutputTokens: 0 },
                    modelContextWindow: 128000,
                }),
                createTokenUsageNotification(sessionId, {
                    total: { totalTokens: 2000, inputTokens: 1600, cachedInputTokens: 0, cacheWriteInputTokens: 0, outputTokens: 400, reasoningOutputTokens: 0 },
                    last: { totalTokens: 1000, inputTokens: 800, cachedInputTokens: 0, cacheWriteInputTokens: 0, outputTokens: 200, reasoningOutputTokens: 0 },
                    modelContextWindow: 128000,
                }),
                createTokenUsageNotification(sessionId, {
                    total: { totalTokens: 3500, inputTokens: 2800, cachedInputTokens: 500, cacheWriteInputTokens: 0, outputTokens: 600, reasoningOutputTokens: 100 },
                    last: { totalTokens: 1500, inputTokens: 1200, cachedInputTokens: 500, cacheWriteInputTokens: 0, outputTokens: 200, reasoningOutputTokens: 100 },
                    modelContextWindow: 128000,
                }),
            ])();

            await expect(`${JSON.stringify(events, null, 2)}\n`).toMatchFileSnapshot('data/token-usage-session-update-multiple.json');
        });

        it('should skip usage_update when model context window is unavailable', async () => {
            const events = await setupPromptAndReturnEvents([
                createTokenUsageNotification(sessionId, {
                    total: { totalTokens: 5000, inputTokens: 4000, cachedInputTokens: 1000, cacheWriteInputTokens: 0, outputTokens: 900, reasoningOutputTokens: 100 },
                    last: { totalTokens: 2500, inputTokens: 2000, cachedInputTokens: 500, cacheWriteInputTokens: 0, outputTokens: 450, reasoningOutputTokens: 50 },
                    modelContextWindow: null,
                }),
            ])();

            expect(events).toEqual([]);
        });
    });
    describe('usage of child threads', () => {
        const childThreadId = 'child-thread-id';
        const parentUsage = createTokenUsageNotification(sessionId, {
            total: breakdown(1000, 800, 0, 200),
            last: breakdown(1000, 800, 0, 200),
            modelContextWindow: 128000,
        });
        const childUsage = createTokenUsageNotification(childThreadId, {
            total: breakdown(50000, 49000, 40000, 1000),
            last: breakdown(50000, 49000, 40000, 1000),
            modelContextWindow: 200000,
        }, 'child-turn-id');
        const childSpawned: ServerNotification = {
            method: 'item/started',
            params: {
                threadId: sessionId,
                turnId: 'turn-id',
                startedAtMs: 0,
                item: {
                    type: 'subAgentActivity',
                    model: null,
                    reasoningEffort: null,
                    id: 'activity-started',
                    kind: 'started',
                    agentThreadId: childThreadId,
                    agentPath: '/root/worker',
                },
            },
        };

        const childCompleted: ServerNotification = {
            method: 'turn/completed',
            params: {
                threadId: childThreadId,
                turn: {
                    id: 'child-turn-id',
                    items: [],
                    rootTurnId: null,
                    itemsView: 'notLoaded',
                    status: 'completed',
                    error: null,
                    startedAt: null,
                    completedAt: null,
                    durationMs: null,
                },
            },
        };

        it('keeps the usage of a native subagent out of the parent session', async () => {
            const codexAcpAgent = mockFixture.getCodexAcpAgent();
            await codexAcpAgent.initialize({
                protocolVersion: 1,
                clientCapabilities: {
                    _meta: { jetbrains: { air: { version: 1, capabilities: ['nativeSubagentSessions'] } } },
                },
            });
            const sessionState = createTestSessionState({ sessionId });
            sessionState.subagents = new CodexSubagentEventRouter(
                sessionId,
                true,
                new ACPSessionConnection(mockFixture.getAcpConnection(), sessionId),
                () => {},
                () => ({currentModel: null, availableModels: []}),
            );
            mockFixture.getCodexAppServerClient().turnStart = vi.fn().mockResolvedValue({
                turn: { id: 'turn-id', items: [], status: 'inProgress', error: null },
            });
            mockFixture.getCodexAppServerClient().awaitTurnCompleted = vi.fn().mockImplementation(async () => {
                for (const notification of [childSpawned, reasoningStarted(sessionId), parentUsage, childUsage, childCompleted]) {
                    mockFixture.sendServerNotification(notification);
                }
                return {
                    threadId: sessionId,
                    turn: { id: 'turn-id', items: [], status: 'completed', error: null },
                };
            });
            vi.spyOn(codexAcpAgent, 'getSessionState').mockReturnValue(sessionState);

            const response = await codexAcpAgent.prompt({
                sessionId,
                prompt: [{ type: 'text', text: 'test prompt' }],
            });

            const usageUpdates = mockFixture.getAcpConnectionEvents([])
                .filter(event => event.method === 'sessionUpdate' && event.args[0].update.sessionUpdate === 'usage_update')
                .map(event => ({ sessionId: event.args[0].sessionId, used: event.args[0].update.used, size: event.args[0].update.size }));
            expect(usageUpdates).toEqual([
                { sessionId, used: 1000, size: 128000 },
                { sessionId: childThreadId, used: 50000, size: 200000 },
            ]);
            expect(sessionState.lastTokenUsage?.totalTokens).toBe(1000);
            expect(sessionState.totalTokenUsage?.totalTokens).toBe(1000);
            expect(sessionState.modelContextWindow).toBe(128000);
            expect(response.usage?.totalTokens).toBe(1000);
        });

    });

    describe('usage records', () => {
        const usageMeta = (events: any[]) => events
            .filter(event => event.method === 'sessionUpdate' && event.args[0].update.sessionUpdate === 'session_info_update'
                && event.args[0].update._meta?.['workbench/usage'])
            .map(event => ({ sessionId: event.args[0].sessionId, ...event.args[0].update._meta['workbench/usage'] }));
        const usageUpdates = (events: any[]) => events
            .filter(event => event.method === 'sessionUpdate' && event.args[0].update.sessionUpdate === 'usage_update');
        const fastModel = { id: 'child-model', additionalSpeedTiers: ['fast'] } as unknown as Model;
        const request = (total: number, last: number) => createTokenUsageNotification(sessionId, {
            total: { totalTokens: total, inputTokens: total, cachedInputTokens: 0, cacheWriteInputTokens: 0, outputTokens: 0, reasoningOutputTokens: 0 },
            last: { totalTokens: last, inputTokens: last, cachedInputTokens: 0, cacheWriteInputTokens: 0, outputTokens: 0, reasoningOutputTokens: 0 },
            modelContextWindow: 128000,
        });

        function newHandler(overrides: Partial<SessionState> = {}) {
            const sessionState = createTestSessionState({ sessionId, authProvider: 'azure', ...overrides });
            sessionState.promptTokenUsage = new PromptTokenUsage(null);
            sessionState.promptTokenUsage.observeModelOutput();
            return { sessionState, handler: createTestEventHandler(mockFixture.getAcpConnection(), sessionState) };
        }

        it('sends one request record per counted update, with model, provider and tier', async () => {
            const { handler } = newHandler({ fastModeEnabled: true, currentModelSupportsFast: true });

            await handler.handleNotification(createTokenUsageNotification(sessionId, {
                total: { totalTokens: 5000, inputTokens: 4000, cachedInputTokens: 1000, cacheWriteInputTokens: 200, outputTokens: 900, reasoningOutputTokens: 100 },
                last: { totalTokens: 2500, inputTokens: 2000, cachedInputTokens: 500, cacheWriteInputTokens: 100, outputTokens: 450, reasoningOutputTokens: 50 },
                modelContextWindow: 128000,
            }));

            expect(usageMeta(mockFixture.getAcpConnectionEvents([]))).toEqual([{
                sessionId, scope: 'request', provider: 'azure', model: 'model-id',
                input: 1400, cacheRead: 500, cacheWrite: 100, output: 450, reasoning: 50, serviceTier: 'fast',
            }]);
        });

        it('sends no record for a repeated update, the first update after a resume or an all-zero update', async () => {
            const { sessionState, handler } = newHandler();
            sessionState.promptTokenUsage = new PromptTokenUsage(null);

            await handler.handleNotification(request(9000, 500));
            sessionState.promptTokenUsage.observeModelOutput();
            await handler.handleNotification(request(9500, 500));
            await handler.handleNotification(request(9500, 500));
            await handler.handleNotification(createTokenUsageNotification(sessionId, {
                total: { totalTokens: 128000, inputTokens: 0, cachedInputTokens: 0, cacheWriteInputTokens: 0, outputTokens: 0, reasoningOutputTokens: 0 },
                last: { totalTokens: 128000, inputTokens: 0, cachedInputTokens: 0, cacheWriteInputTokens: 0, outputTokens: 0, reasoningOutputTokens: 0 },
                modelContextWindow: 128000,
            }));

            expect(usageMeta(mockFixture.getAcpConnectionEvents([])).map(record => record.input)).toEqual([500]);
        });

        it('adds up to the usage of the prompt answer', async () => {
            const codexAcpAgent = mockFixture.getCodexAcpAgent();
            mockFixture.getCodexAppServerClient().turnStart = vi.fn().mockResolvedValue({
                turn: { id: 'turn-id', items: [], status: 'inProgress', error: null },
            });
            mockFixture.getCodexAppServerClient().awaitTurnCompleted = vi.fn().mockImplementation(async () => {
                for (const notification of [reasoningStarted(sessionId), request(1000, 1000), request(1000, 1000), request(2200, 1200), request(3500, 1300)]) {
                    mockFixture.sendServerNotification(notification);
                }
                return { threadId: sessionId, turn: { id: 'turn-id', items: [], status: 'completed', error: null } };
            });
            vi.spyOn(codexAcpAgent, 'getSessionState').mockReturnValue(createTestSessionState({ sessionId }));

            const response = await codexAcpAgent.prompt({ sessionId, prompt: [{ type: 'text', text: 'test prompt' }] });

            const records = usageMeta(mockFixture.getAcpConnectionEvents([]));
            expect(records).toHaveLength(3);
            expect(records.reduce((sum, record) => sum + record.input, 0)).toBe(response.usage?.inputTokens);
        });

        describe('hidden subagents', () => {
            const childThreadId = 'hidden-child';
            const childStarted = (model: string): ServerNotification => ({
                method: 'thread/started',
                params: { thread: { id: childThreadId, model } as any },
            });
            const childUsage = (total: number, last: number) => createTokenUsageNotification(childThreadId, {
                total: { totalTokens: total, inputTokens: total, cachedInputTokens: 0, cacheWriteInputTokens: 0, outputTokens: 0, reasoningOutputTokens: 0 },
                last: { totalTokens: last, inputTokens: last, cachedInputTokens: 0, cacheWriteInputTokens: 0, outputTokens: 0, reasoningOutputTokens: 0 },
                modelContextWindow: 200000,
            }, 'child-turn-id');

            it('sends one subagent record on the root session and leaves the root usage alone', async () => {
                const { sessionState, handler } = newHandler({ fastModeEnabled: true, availableModels: [fastModel] });

                await handler.handleNotification(childStarted('child-model'));
                await handler.handleNotification(childUsage(7000, 7000));
                await handler.handleNotification(childUsage(7000, 7000));
                await handler.handleNotification(childUsage(9000, 2000));

                const events = mockFixture.getAcpConnectionEvents([]);
                expect(usageMeta(events)).toEqual([
                    expect.objectContaining({ sessionId, scope: 'subagent', model: 'child-model', input: 7000, serviceTier: 'fast' }),
                    expect.objectContaining({ sessionId, scope: 'subagent', model: 'child-model', input: 2000, serviceTier: 'fast' }),
                ]);
                expect(usageUpdates(events)).toEqual([]);
                expect(sessionState.lastTokenUsage).toBeNull();
                expect(sessionState.totalTokenUsage).toBeNull();
                expect(sessionState.promptTokenUsage?.usage()).toBeNull();
                expect(sessionState.modelContextWindow).toBeNull();
            });

            it('uses the root model, and no tier, for a child whose model was not seen', async () => {
                const { handler } = newHandler({ fastModeEnabled: true });

                await handler.handleNotification(childUsage(100, 100));

                const [record] = usageMeta(mockFixture.getAcpConnectionEvents([]));
                expect(record).toMatchObject({ scope: 'subagent', model: 'model-id' });
                expect(record).not.toHaveProperty('serviceTier');
            });

            it('sends the record of an update that arrives after the turn ended', async () => {
                const { handler } = newHandler();

                await handler.handleSessionScopedNotification(childUsage(100, 100));

                expect(usageMeta(mockFixture.getAcpConnectionEvents([]))).toHaveLength(1);
            });

            it('reaches the root handler from a child discovered during a prompt', async () => {
                const codexAcpAgent = mockFixture.getCodexAcpAgent();
                mockFixture.getCodexAppServerClient().turnStart = vi.fn().mockResolvedValue({
                    turn: { id: 'turn-id', items: [], status: 'inProgress', error: null },
                });
                const spawned: ServerNotification = {
                    method: 'item/started',
                    params: {
                        threadId: sessionId, turnId: 'turn-id', startedAtMs: 0,
                        item: { type: 'subAgentActivity', model: null, reasoningEffort: null, id: 'activity', kind: 'started', agentThreadId: childThreadId, agentPath: '/root/worker' },
                    },
                };
                mockFixture.getCodexAppServerClient().awaitTurnCompleted = vi.fn().mockImplementation(async () => {
                    for (const notification of [spawned, childUsage(300, 300)]) {
                        mockFixture.sendServerNotification(notification);
                    }
                    return { threadId: sessionId, turn: { id: 'turn-id', items: [], status: 'completed', error: null } };
                });
                vi.spyOn(codexAcpAgent, 'getSessionState').mockReturnValue(createTestSessionState({ sessionId }));

                await codexAcpAgent.prompt({ sessionId, prompt: [{ type: 'text', text: 'test prompt' }] });

                expect(usageMeta(mockFixture.getAcpConnectionEvents([]))).toEqual([
                    expect.objectContaining({ sessionId, scope: 'subagent', input: 300 }),
                ]);
            });
        });
    });
});
