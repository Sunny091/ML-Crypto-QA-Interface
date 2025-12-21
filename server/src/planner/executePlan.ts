/**
 * Plan Executor / Router
 *
 * Phase 4: Routes Orchestrator JSON plans to fastmcp tool calls.
 *
 * Responsibilities:
 * - Validate if a plan is executable
 * - Convert plan to tool input format
 * - Route to correct tool (price-only or price+text)
 * - Return structured result
 *
 * Does NOT:
 * - Understand natural language
 * - Make predictions itself
 * - Modify tool output values
 */

import { Plan } from '../schemas/plan.js';
import {
  callPredictPricePriceOnly,
  callPredictPriceWithText,
  PredictPriceResult,
  PredictPriceWithTextResult,
  MCPClientError,
} from '../services/mcpClient.js';
import { config } from '../config/env.js';

/**
 * Execution result when tool is called successfully
 */
export interface ExecutionSuccess {
  executed: true;
  tool_called: true;
  tool_name: string;
  tool_result: PredictPriceResult | PredictPriceWithTextResult;
  assistant_text: string;
}

/**
 * Execution result when tool is not called
 */
export interface ExecutionSkipped {
  executed: false;
  tool_called: false;
  reason: string;
  assistant_text: string;
}

/**
 * Execution result when tool call fails
 */
export interface ExecutionError {
  executed: false;
  tool_called: false;
  error: string;
  details?: string;
  assistant_text: string;
}

export type ExecutionResult = ExecutionSuccess | ExecutionSkipped | ExecutionError;

/**
 * Check if a plan can be executed with current Phase 4 capabilities
 *
 * Phase 4 supports:
 * - task === "predict"
 * - symbol !== null
 * - use_text_features === false (price-only)
 * - use_text_features === true with news_text (price+text)
 */
function canExecutePlan(plan: Plan): { canExecute: boolean; reason?: string } {
  // Only "predict" task is supported
  if (plan.task !== 'predict') {
    return {
      canExecute: false,
      reason: `Task "${plan.task}" is not executable. Only "predict" task is supported.`,
    };
  }

  // Symbol must be specified
  if (plan.symbol === null) {
    return {
      canExecute: false,
      reason: 'No symbol specified. Please specify BTC, ETH, or SOL.',
    };
  }

  // If text features requested, news_text must be provided
  if (plan.use_text_features && (!plan.news_text || plan.news_text.trim() === '')) {
    return {
      canExecute: false,
      reason: 'Text features requested but no news_text provided. Please include news text for sentiment analysis.',
    };
  }

  return { canExecute: true };
}

/**
 * Convert plan to price-only tool input format
 */
function planToPriceOnlyInput(plan: Plan): { symbol: 'BTC' | 'ETH' | 'SOL'; as_of: string } {
  // Resolve "now" to today's date
  let as_of: string;
  if (plan.as_of === null || plan.as_of === 'now') {
    as_of = new Date().toISOString().split('T')[0]; // YYYY-MM-DD
  } else {
    as_of = plan.as_of;
  }

  return {
    symbol: plan.symbol as 'BTC' | 'ETH' | 'SOL',
    as_of,
  };
}

/**
 * Convert plan to price+text tool input format (Phase 4)
 */
function planToPriceTextInput(plan: Plan): { symbol: 'BTC' | 'ETH' | 'SOL'; as_of: string; news_text: string } {
  const base = planToPriceOnlyInput(plan);
  return {
    ...base,
    news_text: plan.news_text!,  // Already validated in canExecutePlan
  };
}

/**
 * Generate user-facing message based on tool result
 */
function generateResultMessage(result: PredictPriceResult | PredictPriceWithTextResult): string {
  const direction = result.prediction === 'UP' ? '上漲' : '下跌';
  const confidence = (result.prob_up * 100).toFixed(1);

  // Check if this is a price+text result (has sentiment_score)
  if ('sentiment_score' in result) {
    const sentimentDesc = result.sentiment_score > 0.2 ? '正面'
      : result.sentiment_score < -0.2 ? '負面' : '中性';
    const sentimentValue = result.sentiment_score.toFixed(2);

    return `使用價格+新聞模型預測 ${result.symbol}：預測隔日${direction}（信心度 ${confidence}%）。新聞情緒分析：${sentimentDesc}（${sentimentValue}）。`;
  }

  return `使用價格模型預測 ${result.symbol}：預測隔日${direction}（信心度 ${confidence}%）。模型僅使用價格特徵，未使用新聞資料。`;
}

/**
 * Execute a plan by routing to appropriate tool
 *
 * @param plan Orchestrator JSON plan
 * @returns Execution result
 */
export async function executePlan(plan: Plan): Promise<ExecutionResult> {
  if (config.app.isDev) {
    console.log('[Planner] Executing plan:', JSON.stringify(plan));
  }

  // Check if plan can be executed
  const { canExecute, reason } = canExecutePlan(plan);

  if (!canExecute) {
    if (config.app.isDev) {
      console.log(`[Planner] Plan not executable: ${reason}`);
    }

    return {
      executed: false,
      tool_called: false,
      reason: reason!,
      assistant_text: reason!,
    };
  }

  // Route to appropriate tool based on use_text_features
  try {
    if (plan.use_text_features) {
      // Phase 4: Use price+text model
      const toolInput = planToPriceTextInput(plan);

      if (config.app.isDev) {
        console.log('[Planner] Using price+text tool');
        console.log('[Planner] Tool input:', JSON.stringify({
          ...toolInput,
          news_text: toolInput.news_text.substring(0, 50) + '...',
        }));
      }

      const toolResult = await callPredictPriceWithText(toolInput);

      return {
        executed: true,
        tool_called: true,
        tool_name: 'predict_price_with_text',
        tool_result: toolResult,
        assistant_text: generateResultMessage(toolResult),
      };
    } else {
      // Phase 2/3: Use price-only model
      const toolInput = planToPriceOnlyInput(plan);

      if (config.app.isDev) {
        console.log('[Planner] Using price-only tool');
        console.log('[Planner] Tool input:', JSON.stringify(toolInput));
      }

      const toolResult = await callPredictPricePriceOnly(toolInput);

      return {
        executed: true,
        tool_called: true,
        tool_name: 'predict_price_price_only',
        tool_result: toolResult,
        assistant_text: generateResultMessage(toolResult),
      };
    }
  } catch (error) {
    const errorMessage =
      error instanceof MCPClientError
        ? error.message
        : 'Unknown error calling tool';

    const errorDetails =
      error instanceof MCPClientError ? error.details : undefined;

    if (config.app.isDev) {
      console.log(`[Planner] Tool call failed: ${errorMessage}`);
      if (errorDetails) console.log(`[Planner] Details: ${errorDetails}`);
    }

    return {
      executed: false,
      tool_called: false,
      error: errorMessage,
      details: errorDetails,
      assistant_text: `無法執行預測：${errorMessage}`,
    };
  }
}
