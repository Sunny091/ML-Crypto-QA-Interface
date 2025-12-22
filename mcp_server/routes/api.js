// routes/api.js - REST API 路由
import express from 'express';
import { Logger } from '../utils/logger.js';

export function createAPIRoutes(mcpClient) {
  const router = express.Router();

  // 健康檢查
  router.get('/health', (req, res) => {
    const tools = mcpClient.getTools();
    res.json({
      status: 'ok',
      tools: tools.length,
      timestamp: new Date().toISOString()
    });
  });

  // 獲取工具列表
  router.get('/tools', (req, res) => {
    const tools = mcpClient.getTools();
    res.json({ tools });
  });

  // 聊天 API
  router.post('/chat', async (req, res) => {
    try {
      const { message, history = [] } = req.body;

      if (!message) {
        return res.status(400).json({ error: '缺少 message 參數' });
      }

      Logger.info(`收到聊天請求: ${message.substring(0, 50)}...`);

      const result = await mcpClient.chat(message, history);

      // 提取文字回應
      const textContent = result.response.content
        .filter(c => c.type === 'text')
        .map(c => c.text)
        .join('\n');

      res.json({
        success: true,
        response: textContent,
        raw: result.response,
        usage: result.response.usage
      });

    } catch (error) {
      Logger.error('聊天請求失敗:', error.message);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  // 即時價格查詢
  router.get('/price/:symbol', async (req, res) => {
    try {
      const symbol = req.params.symbol.toUpperCase();

      if (!['BTC', 'ETH', 'SOL'].includes(symbol)) {
        return res.status(400).json({ error: '不支援的幣種，請使用 BTC, ETH 或 SOL' });
      }

      Logger.info(`查詢即時價格: ${symbol}`);

      // 呼叫 MCP 工具 (FastMCP 需要 input 包裝)
      const result = await mcpClient.callTool('get_current_price', { input: { symbol } });

      res.json(result);

    } catch (error) {
      Logger.error('價格查詢失敗:', error.message);
      res.status(500).json({ error: error.message });
    }
  });

  // 圖表數據 (給 Chart.js 用)
  router.post('/chart-data', async (req, res) => {
    try {
      const { symbol = 'BTC', days, interval = 'd1', start_date, end_date } = req.body;

      const upperSymbol = symbol.toUpperCase();
      if (!['BTC', 'ETH', 'SOL'].includes(upperSymbol)) {
        return res.status(400).json({ error: '不支援的幣種' });
      }

      // 構建請求參數
      const input = {
        symbol: upperSymbol,
        interval
      };

      // 優先使用日期範圍，否則使用天數
      if (start_date && end_date) {
        input.start_date = start_date;
        input.end_date = end_date;
        Logger.info(`查詢圖表數據: ${upperSymbol}, ${start_date} ~ ${end_date}`);
      } else {
        input.days = Math.min(Math.max(parseInt(days) || 30, 1), 365);
        Logger.info(`查詢圖表數據: ${upperSymbol}, ${input.days} 天`);
      }

      // 呼叫 MCP 工具取得歷史價格 (FastMCP 需要 input 包裝)
      const result = await mcpClient.callTool('get_price_history', { input });

      // 轉換為 Chart.js 格式
      const timestamps = result.data.map(p => p.timestamp);
      const prices = result.data.map(p => p.price_usd);

      res.json({
        symbol: result.symbol,
        timestamps,
        prices,
        start_date: result.start_date,
        end_date: result.end_date
      });

    } catch (error) {
      Logger.error('圖表數據查詢失敗:', error.message);
      res.status(500).json({ error: error.message });
    }
  });

  // 聊天 API (SSE 串流)
  router.post('/chat/stream', async (req, res) => {
    try {
      const { message, history = [] } = req.body;

      if (!message) {
        return res.status(400).json({ error: '缺少 message 參數' });
      }

      // 設置 SSE
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      Logger.info(`收到串流聊天請求: ${message.substring(0, 50)}...`);

      // 發送處理狀態
      res.write(`data: ${JSON.stringify({ type: 'status', message: '正在處理...' })}\n\n`);

      const result = await mcpClient.chat(message, history);

      // 發送結果
      const textContent = result.response.content
        .filter(c => c.type === 'text')
        .map(c => c.text)
        .join('\n');

      res.write(`data: ${JSON.stringify({ type: 'content', text: textContent })}\n\n`);
      res.write(`data: ${JSON.stringify({ type: 'done', usage: result.response.usage })}\n\n`);
      res.end();

    } catch (error) {
      Logger.error('串流聊天請求失敗:', error.message);
      res.write(`data: ${JSON.stringify({ type: 'error', message: error.message })}\n\n`);
      res.end();
    }
  });

  return router;
}
