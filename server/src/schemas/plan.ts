import { z } from 'zod';

/**
 * JSON Schema for LLM orchestrator output
 * This is the contract between the LLM and the system
 */
export const planSchema = z.object({
  task: z.enum(['predict', 'analyze_news', 'help', 'unknown']),
  symbol: z.enum(['BTC', 'ETH', 'SOL']).nullable(),
  as_of: z.union([
    z.literal('now'),
    z.string().regex(/^\d{4}-\d{2}-\d{2}$/, 'Date must be YYYY-MM-DD format'),
  ]).nullable(),
  horizon: z.enum(['1d', '7d']).nullable(),
  use_text_features: z.boolean(),
  news_text_provided: z.boolean(),
  news_text: z.string().nullable(),
});

export type Plan = z.infer<typeof planSchema>;

/**
 * Validate and parse LLM output as Plan
 * Returns { success: true, data: Plan } or { success: false, error: string }
 */
export function validatePlan(input: unknown):
  | { success: true; data: Plan }
  | { success: false; error: string } {
  const result = planSchema.safeParse(input);

  if (result.success) {
    return { success: true, data: result.data };
  }

  return {
    success: false,
    error: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`).join('; ')
  };
}

/**
 * JSON Schema representation for inclusion in system prompt
 */
export const planSchemaDescription = `{
  "task": "predict" | "analyze_news" | "help" | "unknown",
  "symbol": "BTC" | "ETH" | "SOL" | null,
  "as_of": "now" | "YYYY-MM-DD" | null,
  "horizon": "1d" | "7d" | null,
  "use_text_features": true | false,
  "news_text_provided": true | false,
  "news_text": "string | null"
}`;
