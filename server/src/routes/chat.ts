import { Router, Request, Response } from 'express';
import { z } from 'zod';
import { orchestrate } from '../services/ollama.js';
import { config } from '../config/env.js';
import { executePlan, ExecutionResult } from '../planner/executePlan.js';
import { Plan } from '../schemas/plan.js';

const router = Router();

/**
 * Request body schema
 */
const chatRequestSchema = z.object({
  message: z.string().min(1, 'Message is required').max(10000, 'Message too long'),
});

/**
 * Generate assistant text for non-executable plans
 * This is a brief explanation, NOT a prediction
 */
function generateFallbackText(plan: Plan): string {
  switch (plan.task) {
    case 'help':
      return '這是一個加密貨幣分析系統。您可以詢問 BTC、ETH、SOL 的隔日漲跌預測。目前僅支援價格特徵預測，新聞分析功能尚未開放。';
    case 'unknown':
      return '無法理解您的請求。請嘗試詢問加密貨幣的預測，例如：「幫我預測 BTC 現在的隔日漲跌」';
    default:
      return '計畫已生成。';
  }
}

/**
 * POST /api/chat
 *
 * Phase 3 Response Format:
 * {
 *   "plan": { ...JSON schema... },
 *   "tool_called": boolean,
 *   "tool_name": string | undefined,
 *   "tool_result": { ...tool output... } | undefined,
 *   "assistant_text": string
 * }
 */
router.post('/chat', async (req: Request, res: Response) => {
  // Validate request body
  const parseResult = chatRequestSchema.safeParse(req.body);
  if (!parseResult.success) {
    return res.status(400).json({
      error: 'Invalid request',
      details: parseResult.error.errors.map(e => e.message),
    });
  }

  const { message } = parseResult.data;

  if (config.app.isDev) {
    console.log(`[Chat] Received message: ${message.slice(0, 100)}...`);
  }

  // Step 1: Call orchestrator to get JSON plan
  const orchestratorResult = await orchestrate(message);

  if (!orchestratorResult.success) {
    // Orchestrator failed - return error
    const errorResponse: {
      error: string;
      details: string;
      rawResponse?: string;
      retryCount?: number;
    } = {
      error: 'Failed to generate plan',
      details: orchestratorResult.error,
    };

    if (config.app.isDev) {
      errorResponse.rawResponse = orchestratorResult.rawResponse;
      errorResponse.retryCount = orchestratorResult.retryCount;
    }

    return res.status(500).json(errorResponse);
  }

  const plan = orchestratorResult.plan;

  if (config.app.isDev) {
    console.log(`[Chat] Plan generated: ${JSON.stringify(plan)}`);
  }

  // Step 2: Execute plan via planner/router
  const executionResult = await executePlan(plan);

  if (config.app.isDev) {
    console.log(`[Chat] Execution result: ${JSON.stringify(executionResult)}`);
  }

  // Step 3: Build response based on execution result
  if (executionResult.executed && executionResult.tool_called) {
    // Tool was called successfully
    return res.json({
      plan,
      tool_called: true,
      tool_name: executionResult.tool_name,
      tool_result: executionResult.tool_result,
      assistant_text: executionResult.assistant_text,
    });
  }

  // Tool was not called (plan not executable or error)
  if ('reason' in executionResult) {
    // Plan was not executable
    return res.json({
      plan,
      tool_called: false,
      reason: executionResult.reason,
      assistant_text: executionResult.assistant_text || generateFallbackText(plan),
    });
  }

  if ('error' in executionResult) {
    // Tool call failed
    return res.json({
      plan,
      tool_called: false,
      error: executionResult.error,
      details: executionResult.details,
      assistant_text: executionResult.assistant_text,
    });
  }

  // Fallback (should not reach here)
  return res.json({
    plan,
    tool_called: false,
    assistant_text: generateFallbackText(plan),
  });
});

/**
 * GET /api/health
 * Health check endpoint
 */
router.get('/health', (_req: Request, res: Response) => {
  res.json({
    status: 'ok',
    model: config.ollama.model,
    env: config.app.env,
  });
});

export default router;
