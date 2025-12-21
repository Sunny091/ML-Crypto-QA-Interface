import express from 'express';
import cors from 'cors';
import { config } from './config/env.js';
import chatRouter from './routes/chat.js';

const app = express();

// Middleware
app.use(cors());
app.use(express.json({ limit: '1mb' }));

// Request logging in development
if (config.app.isDev) {
  app.use((req, _res, next) => {
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
    next();
  });
}

// Routes
app.use('/api', chatRouter);

// Root endpoint
app.get('/', (_req, res) => {
  res.json({
    name: 'MCP Orchestrator Server',
    version: '1.0.0',
    phase: 1,
    description: 'Control plane for natural language cryptocurrency analysis',
    endpoints: {
      chat: 'POST /api/chat',
      health: 'GET /api/health',
    },
  });
});

// Error handling middleware
app.use((err: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  console.error('Unhandled error:', err);
  res.status(500).json({
    error: 'Internal server error',
    message: config.app.isDev ? err.message : undefined,
  });
});

// Start server
const server = app.listen(config.server.port, config.server.host, () => {
  console.log('');
  console.log('╔════════════════════════════════════════════════════════╗');
  console.log('║         MCP Orchestrator Server - Phase 1              ║');
  console.log('╠════════════════════════════════════════════════════════╣');
  console.log(`║  URL:    http://${config.server.host}:${config.server.port}`.padEnd(59) + '║');
  console.log(`║  Model:  ${config.ollama.model}`.padEnd(59) + '║');
  console.log(`║  Env:    ${config.app.env}`.padEnd(59) + '║');
  console.log('╚════════════════════════════════════════════════════════╝');
  console.log('');
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully...');
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down gracefully...');
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});
