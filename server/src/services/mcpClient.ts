/**
 * MCP Client Service
 *
 * Phase 3: Calls fastmcp tool server via HTTP.
 * Backend does NOT compute predictions itself - it only routes to the tool server.
 */

import { config } from '../config/env.js';

/**
 * Tool result schema (matches mcp_server output)
 */
export interface PredictPriceResult {
  symbol: string;
  as_of: string;
  prediction: 'UP' | 'DOWN';
  prob_up: number;
  model_variant: string;
  features_used: string[];
}

/**
 * Phase 4: Price+text tool result schema
 */
export interface PredictPriceWithTextResult extends PredictPriceResult {
  sentiment_score: number;
}

/**
 * Tool input schema
 */
export interface PredictPriceInput {
  symbol: 'BTC' | 'ETH' | 'SOL';
  as_of: string; // YYYY-MM-DD format
}

/**
 * Phase 4: Price+text tool input schema
 */
export interface PredictPriceWithTextInput extends PredictPriceInput {
  news_text: string;
}

/**
 * MCP client error
 */
export class MCPClientError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public details?: string
  ) {
    super(message);
    this.name = 'MCPClientError';
  }
}

/**
 * Check if MCP tool server is healthy
 */
export async function checkMCPHealth(): Promise<{
  status: string;
  server: string;
  tools: string[];
}> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config.mcpClient.timeoutMs);

  try {
    const response = await fetch(`${config.mcpClient.host}/health`, {
      method: 'GET',
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new MCPClientError(
        'MCP health check failed',
        response.status,
        await response.text()
      );
    }

    return response.json() as Promise<{ status: string; server: string; tools: string[] }>;
  } catch (error) {
    if (error instanceof MCPClientError) throw error;
    throw new MCPClientError(
      'Failed to connect to MCP server',
      undefined,
      error instanceof Error ? error.message : 'Unknown error'
    );
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Call predict_price_price_only tool
 *
 * @param input Tool input (symbol and as_of date)
 * @returns Tool result with prediction
 */
export async function callPredictPricePriceOnly(
  input: PredictPriceInput
): Promise<PredictPriceResult> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config.mcpClient.timeoutMs);

  const url = `${config.mcpClient.host}/tools/predict_price_price_only`;

  if (config.app.isDev) {
    console.log(`[MCPClient] Calling ${url}`);
    console.log(`[MCPClient] Input: ${JSON.stringify(input)}`);
  }

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(input),
      signal: controller.signal,
    });

    const data = await response.json() as PredictPriceResult & { error?: string; details?: string };

    if (!response.ok) {
      throw new MCPClientError(
        data.error || 'Tool execution failed',
        response.status,
        data.details
      );
    }

    if (config.app.isDev) {
      console.log(`[MCPClient] Result: ${JSON.stringify(data)}`);
    }

    return data;
  } catch (error) {
    if (error instanceof MCPClientError) throw error;

    // Handle network errors
    const message = error instanceof Error ? error.message : 'Unknown error';
    throw new MCPClientError(
      'Failed to call MCP tool',
      undefined,
      message.includes('ECONNREFUSED')
        ? 'MCP server is not running. Start it with: cd mcp_server && python http_server.py'
        : message
    );
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Call predict_price_with_text tool (Phase 4)
 *
 * @param input Tool input (symbol, as_of date, and news_text)
 * @returns Tool result with prediction and sentiment_score
 */
export async function callPredictPriceWithText(
  input: PredictPriceWithTextInput
): Promise<PredictPriceWithTextResult> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config.mcpClient.timeoutMs);

  const url = `${config.mcpClient.host}/tools/predict_price_with_text`;

  if (config.app.isDev) {
    console.log(`[MCPClient] Calling ${url}`);
    console.log(`[MCPClient] Input: symbol=${input.symbol}, as_of=${input.as_of}, news_text=${input.news_text.substring(0, 50)}...`);
  }

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(input),
      signal: controller.signal,
    });

    const data = await response.json() as PredictPriceWithTextResult & { error?: string; details?: string };

    if (!response.ok) {
      throw new MCPClientError(
        data.error || 'Tool execution failed',
        response.status,
        data.details
      );
    }

    if (config.app.isDev) {
      console.log(`[MCPClient] Result: ${JSON.stringify(data)}`);
    }

    return data;
  } catch (error) {
    if (error instanceof MCPClientError) throw error;

    // Handle network errors
    const message = error instanceof Error ? error.message : 'Unknown error';
    throw new MCPClientError(
      'Failed to call MCP tool',
      undefined,
      message.includes('ECONNREFUSED')
        ? 'MCP server is not running. Start it with: cd mcp_server && python http_server.py'
        : message
    );
  } finally {
    clearTimeout(timeout);
  }
}
