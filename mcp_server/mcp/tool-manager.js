// mcp/tool-manager.js - MCP 工具管理
import { Logger } from '../utils/logger.js';

export class MCPToolManager {
  constructor() {
    this.availableTools = new Map();
    this.serverConnections = new Map();
  }

  /**
   * 註冊伺服器連接
   */
  registerServer(serverName, connection) {
    this.serverConnections.set(serverName, connection);
  }

  /**
   * 註冊工具
   */
  registerTools(serverName, tools) {
    for (const tool of tools) {
      Logger.success(`註冊工具: ${tool.name} (來自 ${serverName})`);
      this.availableTools.set(tool.name, {
        serverName,
        tool,
      });
    }
  }

  /**
   * 獲取 OpenAI 格式的工具定義
   */
  getOpenAITools() {
    const tools = [];

    for (const [toolName, toolInfo] of this.availableTools) {
      tools.push({
        type: 'function',
        function: {
          name: toolName,
          description: toolInfo.tool.description,
          parameters: toolInfo.tool.inputSchema || { type: "object", properties: {} },
        }
      });
    }

    return tools;
  }

  /**
   * 執行工具調用
   */
  async executeTool(toolName, arguments_) {
    const toolInfo = this.availableTools.get(toolName);
    if (!toolInfo) {
      throw new Error(`工具不存在: ${toolName}`);
    }

    const connection = this.serverConnections.get(toolInfo.serverName);
    if (!connection) {
      throw new Error(`伺服器連接不存在: ${toolInfo.serverName}`);
    }

    return await connection.executeToolCall(toolName, arguments_);
  }

  /**
   * 獲取工具資訊
   */
  getToolInfo(toolName) {
    return this.availableTools.get(toolName);
  }

  /**
   * 檢查工具是否存在
   */
  hasTool(toolName) {
    return this.availableTools.has(toolName);
  }

  /**
   * 獲取所有工具名稱
   */
  getToolNames() {
    return Array.from(this.availableTools.keys());
  }

  /**
   * 獲取工具統計
   */
  getStats() {
    return {
      totalTools: this.availableTools.size,
      servers: Array.from(this.serverConnections.keys())
    };
  }

  /**
   * 關閉所有連接
   */
  closeAll() {
    for (const [name, connection] of this.serverConnections) {
      connection.close();
    }
    this.serverConnections.clear();
    this.availableTools.clear();
  }
}
