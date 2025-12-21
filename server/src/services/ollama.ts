import { config } from '../config/env.js';
import { validatePlan, planSchemaDescription, type Plan } from '../schemas/plan.js';

/**
 * System prompt for the orchestrator LLM
 * LLM outputs JSON only - no natural language, no predictions
 */
const SYSTEM_PROMPT = `You are an orchestrator.
Your job is to extract structured intent from user input.
You must output JSON only, strictly following the provided schema.
Do NOT include explanations, natural language, or predictions.
Do NOT wrap the JSON in markdown code blocks.
If the intent is unclear, set task to "unknown".

Schema:
${planSchemaDescription}

Rules:
- If no cryptocurrency symbol is mentioned, set symbol to null
- If no date is mentioned, set as_of to "now"
- If news text is provided in the message, set news_text_provided to true and include the text (max 2000 chars)
- If the user asks about prediction/forecasting, set task to "predict"
- If the user asks to analyze news, set task to "analyze_news"
- If the user asks for help or how to use the system, set task to "help"
- If you cannot understand the intent, set task to "unknown"
- use_text_features should be true if news_text is provided
- horizon should be "1d" for next day predictions, "7d" for weekly`;

const MAX_NEWS_TEXT_LENGTH = 2000;
const MAX_RETRIES = 2;

interface OllamaChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface OllamaChatResponse {
  model: string;
  created_at: string;
  message: OllamaChatMessage;
  done: boolean;
}

/**
 * Extract JSON from LLM response (handles potential markdown wrapping)
 */
function extractJson(text: string): string {
  // Try to extract JSON from markdown code block if present
  const codeBlockMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (codeBlockMatch) {
    return codeBlockMatch[1].trim();
  }

  // Try to find JSON object directly
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (jsonMatch) {
    return jsonMatch[0];
  }

  return text.trim();
}

/**
 * Call Ollama Chat API with the given message
 */
async function callOllama(userMessage: string): Promise<string> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config.ollama.timeoutMs);

  try {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'accept': 'application/json',
    };
    if (config.ollama.apiKey) {
      headers['X-API-Key'] = config.ollama.apiKey;
    }

    const messages: OllamaChatMessage[] = [
      { role: 'system', content: SYSTEM_PROMPT },
      { role: 'user', content: userMessage },
    ];

    const response = await fetch(`${config.ollama.host}/api/chat`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        model: config.ollama.model,
        messages,
        stream: false,
        options: {
          temperature: 0.1, // Low temperature for consistent JSON output
        },
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Ollama API error: ${response.status} - ${errorText}`);
    }

    const data = (await response.json()) as OllamaChatResponse;
    return data.message.content;
  } finally {
    clearTimeout(timeout);
  }
}

export interface OrchestratorResult {
  success: true;
  plan: Plan;
  rawResponse?: string;
}

export interface OrchestratorError {
  success: false;
  error: string;
  rawResponse?: string;
  retryCount: number;
}

export type OrchestratorResponse = OrchestratorResult | OrchestratorError;

/**
 * Call LLM orchestrator and parse response as Plan
 * Implements retry logic for JSON parsing failures
 */
export async function orchestrate(userMessage: string): Promise<OrchestratorResponse> {
  // Truncate any embedded news text if the message is very long
  const truncatedMessage = userMessage.length > MAX_NEWS_TEXT_LENGTH * 2
    ? userMessage.slice(0, MAX_NEWS_TEXT_LENGTH * 2) + '\n[Message truncated]'
    : userMessage;

  let lastError = '';
  let rawResponse = '';

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      // Add retry hint if this is a retry attempt
      const prompt = attempt > 0
        ? `${truncatedMessage}\n\n[Previous response was invalid JSON. Please output ONLY valid JSON matching the schema.]`
        : truncatedMessage;

      rawResponse = await callOllama(prompt);
      const jsonText = extractJson(rawResponse);

      let parsed: unknown;
      try {
        parsed = JSON.parse(jsonText);
      } catch (parseError) {
        lastError = `JSON parse error: ${parseError instanceof Error ? parseError.message : 'Unknown error'}`;
        if (config.app.isDev) {
          console.log(`[Attempt ${attempt + 1}/${MAX_RETRIES + 1}] ${lastError}`);
          console.log(`Raw response: ${rawResponse}`);
        }
        continue;
      }

      // Validate against schema
      const validationResult = validatePlan(parsed);
      if (validationResult.success) {
        // Truncate news_text if present
        if (validationResult.data.news_text && validationResult.data.news_text.length > MAX_NEWS_TEXT_LENGTH) {
          validationResult.data.news_text = validationResult.data.news_text.slice(0, MAX_NEWS_TEXT_LENGTH);
        }
        return {
          success: true,
          plan: validationResult.data,
          rawResponse: config.app.isDev ? rawResponse : undefined,
        };
      }

      lastError = `Schema validation error: ${validationResult.error}`;
      if (config.app.isDev) {
        console.log(`[Attempt ${attempt + 1}/${MAX_RETRIES + 1}] ${lastError}`);
      }
    } catch (error) {
      lastError = error instanceof Error ? error.message : 'Unknown error';
      if (config.app.isDev) {
        console.log(`[Attempt ${attempt + 1}/${MAX_RETRIES + 1}] API error: ${lastError}`);
      }
      // Don't retry on API errors (network issues, etc.)
      break;
    }
  }

  return {
    success: false,
    error: lastError,
    rawResponse: config.app.isDev ? rawResponse : undefined,
    retryCount: MAX_RETRIES,
  };
}
