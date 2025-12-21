import { z } from 'zod';
import dotenv from 'dotenv';
import path from 'path';

// Load .env from project root (one level up from server/)
dotenv.config({ path: path.resolve(__dirname, '../../../.env') });

/**
 * Environment variable schema - single source of truth
 * All process.env reads happen ONLY in this file
 */
const envSchema = z.object({
  // Ollama configuration
  OLLAMA_HOST: z.string().url('OLLAMA_HOST must be a valid URL'),
  OLLAMA_MODEL: z.string().min(1, 'OLLAMA_MODEL is required'),
  OLLAMA_TIMEOUT_MS: z.coerce.number().positive().default(60000),
  OLLAMA_API_KEY: z.string().optional(),

  // Server configuration
  SERVER_HOST: z.string().default('localhost'),
  SERVER_PORT: z.coerce.number().positive().default(3001),

  // App configuration
  APP_ENV: z.enum(['development', 'production', 'test']).default('development'),
  LOG_LEVEL: z.enum(['debug', 'info', 'warn', 'error']).default('info'),

  // Phase 3: MCP Client configuration
  MCP_CLIENT_HOST: z.string().url('MCP_CLIENT_HOST must be a valid URL').default('http://localhost:4000'),
  MCP_CLIENT_TIMEOUT_MS: z.coerce.number().positive().default(30000),
});

type EnvConfig = z.infer<typeof envSchema>;

/**
 * Validate and parse environment variables
 * Throws on startup if required variables are missing
 */
function validateEnv(): EnvConfig {
  const result = envSchema.safeParse({
    OLLAMA_HOST: process.env.OLLAMA_HOST,
    OLLAMA_MODEL: process.env.OLLAMA_MODEL,
    OLLAMA_TIMEOUT_MS: process.env.OLLAMA_TIMEOUT_MS,
    OLLAMA_API_KEY: process.env.OLLAMA_API_KEY,
    SERVER_HOST: process.env.SERVER_HOST,
    SERVER_PORT: process.env.SERVER_PORT,
    APP_ENV: process.env.APP_ENV,
    LOG_LEVEL: process.env.LOG_LEVEL,
    MCP_CLIENT_HOST: process.env.MCP_CLIENT_HOST,
    MCP_CLIENT_TIMEOUT_MS: process.env.MCP_CLIENT_TIMEOUT_MS,
  });

  if (!result.success) {
    console.error('❌ Environment validation failed:');
    console.error(result.error.format());
    process.exit(1);
  }

  return result.data;
}

/**
 * Validated environment configuration
 * Import this instead of reading process.env directly
 */
export const env = validateEnv();

/**
 * Derived configuration for convenience
 */
export const config = {
  ollama: {
    host: env.OLLAMA_HOST,
    model: env.OLLAMA_MODEL,
    timeoutMs: env.OLLAMA_TIMEOUT_MS,
    apiKey: env.OLLAMA_API_KEY,
  },
  server: {
    host: env.SERVER_HOST,
    port: env.SERVER_PORT,
  },
  app: {
    env: env.APP_ENV,
    isDev: env.APP_ENV === 'development',
    isProd: env.APP_ENV === 'production',
    logLevel: env.LOG_LEVEL,
  },
  // Phase 3: MCP Client
  mcpClient: {
    host: env.MCP_CLIENT_HOST,
    timeoutMs: env.MCP_CLIENT_TIMEOUT_MS,
  },
} as const;
