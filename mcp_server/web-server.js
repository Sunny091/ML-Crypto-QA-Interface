// web-server.js - Express 主伺服器
import express from 'express';
import { createServer } from 'http';
import path from 'path';
import { fileURLToPath } from 'url';
import { MCPClient } from './mcp-client.js';
import config from './config/environment.js';
import { Logger } from './utils/logger.js';
import { createAPIRoutes } from './routes/api.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 初始化 Express
const app = express();
const server = createServer(app);

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// 初始化 MCP 客戶端
const mcpClient = new MCPClient();

Logger.server('正在啟動 Crypto Predictor MCP 服務器...');

// 初始化 MCP
await mcpClient.initialize();

// 等待工具載入
setTimeout(() => {
  const tools = mcpClient.getTools();
  Logger.server(`工具載入完成！共 ${tools.length} 個工具可用`);
}, 3000);

// 設定 REST API 路由
app.use('/api', createAPIRoutes(mcpClient));

// 靜態 HTML 頁面
app.get('/', (req, res) => {
  res.sendFile('index.html', { root: path.join(__dirname, 'public') });
});

// 啟動伺服器
const PORT = config.port;
const HOST = config.host;

server.listen(PORT, HOST, () => {
  Logger.server(`====================================`);
  Logger.server(`Crypto Predictor MCP 服務器已啟動`);
  Logger.server(`  URL: http://${HOST}:${PORT}`);
  Logger.server(`  API: http://${HOST}:${PORT}/api`);
  Logger.server(`====================================`);
});

// 優雅關閉
process.on('SIGINT', async () => {
  Logger.server('正在關閉服務器...');
  await mcpClient.cleanup();
  process.exit(0);
});
