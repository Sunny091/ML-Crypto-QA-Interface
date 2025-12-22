// mcp-client.js - MCP 客戶端核心
import { MCPServerConnection } from './mcp/server-connection.js';
import { MCPToolManager } from './mcp/tool-manager.js';
import { OllamaClient } from './api/ollama-client.js';
import { Logger } from './utils/logger.js';
import config from './config/environment.js';

export class MCPClient {
  constructor() {
    this.toolManager = new MCPToolManager();
    this.ollamaClient = new OllamaClient();
  }

  /**
   * 初始化 MCP 客戶端
   */
  async initialize() {
    Logger.server('正在初始化 MCP 客戶端...');

    const servers = config.mcpServers;

    for (const [serverName, serverConfig] of Object.entries(servers)) {
      try {
        Logger.info(`連接到 MCP 服務器: ${serverName}`);
        const connection = new MCPServerConnection(serverName, serverConfig);
        await connection.connect();

        this.toolManager.registerServer(serverName, connection);

        // 等待工具列表載入
        await new Promise(resolve => setTimeout(resolve, 2000));

        const tools = connection.getTools();
        if (tools.length > 0) {
          this.toolManager.registerTools(serverName, tools);
        }
      } catch (error) {
        Logger.error(`連接 ${serverName} 失敗:`, error.message);
      }
    }

    const stats = this.toolManager.getStats();
    Logger.success(`MCP 客戶端初始化完成，共 ${stats.totalTools} 個工具`);
  }

  /**
   * 獲取可用工具列表
   */
  getTools() {
    return this.toolManager.getOpenAITools();
  }

  /**
   * 直接呼叫工具 (用於 API 端點)
   * @param {string} toolName - 工具名稱
   * @param {object} input - 工具輸入參數
   * @returns {object} 工具執行結果
   */
  async callTool(toolName, input) {
    Logger.tool(`直接呼叫工具: ${toolName}`);
    try {
      // 直接傳遞參數，不包裝在 { input: ... } 中
      const result = await this.toolManager.executeTool(toolName, input);
      Logger.success(`工具 ${toolName} 執行成功`);

      // 解析結果 (MCP 工具可能返回多種格式)
      if (Array.isArray(result)) {
        // 如果是 content array，提取 text 內容
        const textContent = result.find(c => c.type === 'text');
        if (textContent && textContent.text) {
          try {
            return JSON.parse(textContent.text);
          } catch {
            return textContent.text;
          }
        }
        return result;
      }

      if (typeof result === 'string') {
        try {
          return JSON.parse(result);
        } catch {
          return result;
        }
      }
      return result;
    } catch (error) {
      Logger.error(`工具 ${toolName} 執行失敗:`, error.message);
      throw error;
    }
  }

  /**
   * 處理聊天請求
   */
  async chat(userMessage, conversationHistory = []) {
    const messages = [
      ...conversationHistory,
      { role: 'user', content: userMessage }
    ];

    const tools = this.getTools();
    let maxIterations = 5;
    let iteration = 0;

    while (iteration < maxIterations) {
      iteration++;
      Logger.info(`對話迭代 ${iteration}/${maxIterations}`);

      const response = await this.ollamaClient.chat(messages, tools);

      // 檢查是否需要執行工具
      const toolUses = response.content.filter(c => c.type === 'tool_use');

      if (toolUses.length === 0) {
        // 沒有工具調用，返回最終結果
        return {
          response: response,
          messages: messages
        };
      }

      // 將助手回應加入對話歷史
      messages.push({
        role: 'assistant',
        content: response.content
      });

      // 執行工具調用
      const toolResults = [];
      for (const toolUse of toolUses) {
        Logger.tool(`執行工具: ${toolUse.name}`);
        try {
          const result = await this.toolManager.executeTool(toolUse.name, toolUse.input);
          toolResults.push({
            type: 'tool_result',
            tool_use_id: toolUse.id,
            content: result
          });
          Logger.success(`工具 ${toolUse.name} 執行成功`);
        } catch (error) {
          Logger.error(`工具 ${toolUse.name} 執行失敗:`, error.message);
          toolResults.push({
            type: 'tool_result',
            tool_use_id: toolUse.id,
            content: [{ type: 'text', text: `錯誤: ${error.message}` }],
            is_error: true
          });
        }
      }

      // 將工具結果加入對話歷史
      messages.push({
        role: 'user',
        content: toolResults
      });
    }

    throw new Error('達到最大迭代次數');
  }

  /**
   * 清理資源
   */
  async cleanup() {
    Logger.info('正在關閉 MCP 客戶端...');
    this.toolManager.closeAll();
    Logger.success('MCP 客戶端已關閉');
  }
}
