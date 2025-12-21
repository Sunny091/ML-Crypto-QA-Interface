import { config } from './config';

/**
 * Plan schema types (matching backend)
 */
export interface Plan {
  task: 'predict' | 'analyze_news' | 'help' | 'unknown';
  symbol: 'BTC' | 'ETH' | 'SOL' | null;
  as_of: string | null;
  horizon: '1d' | '7d' | null;
  use_text_features: boolean;
  news_text_provided: boolean;
  news_text: string | null;
}

/**
 * Tool result schema (Phase 4)
 */
export interface ToolResult {
  symbol: string;
  as_of: string;
  prediction: 'UP' | 'DOWN';
  prob_up: number;
  model_variant: string;
  features_used: string[];
  sentiment_score?: number;  // Phase 4: only present for price+text predictions
}

/**
 * Phase 3 Chat Response
 */
export interface ChatResponse {
  plan: Plan;
  tool_called: boolean;
  tool_name?: string;
  tool_result?: ToolResult;
  reason?: string;
  error?: string;
  details?: string;
  assistant_text: string;
}

export interface ChatError {
  error: string;
  details?: string;
  rawResponse?: string;
}

/**
 * Extended response with timing info
 */
export interface ChatResponseWithTiming extends ChatResponse {
  _timing?: {
    startTime: number;
    endTime: number;
    durationMs: number;
  };
}

/**
 * Send a chat message to the backend
 */
export async function sendChatMessage(message: string): Promise<ChatResponseWithTiming> {
  const startTime = Date.now();

  const response = await fetch(`${config.apiBaseUrl}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message }),
  });

  const endTime = Date.now();

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.details || errorData.error || `HTTP ${response.status}`);
  }

  const data = await response.json();
  return {
    ...data,
    _timing: {
      startTime,
      endTime,
      durationMs: endTime - startTime,
    },
  };
}

/**
 * Check backend health
 */
export async function checkHealth(): Promise<{ status: string; model: string; env: string }> {
  const response = await fetch(`${config.apiBaseUrl}/api/health`);
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`);
  }
  return response.json();
}
