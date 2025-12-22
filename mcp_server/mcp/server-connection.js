// mcp/server-connection.js - MCP 服務器連接管理
import { spawn } from 'child_process';
import { Logger } from '../utils/logger.js';

export class MCPServerConnection {
  constructor(serverName, config) {
    this.serverName = serverName;
    this.config = config;
    this.process = null;
    this.ready = false;
    this.pendingRequests = new Map();
    this.messageBuffer = '';
    this.requestId = 1;
    this.tools = [];
  }

  /**
   * 連接到 MCP 服務器
   */
  async connect() {
    return new Promise((resolve, reject) => {
      Logger.info(`啟動 MCP 服務器: ${this.serverName}`);

      this.process = spawn(this.config.command, this.config.args || [], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env },
        cwd: process.cwd(),
      });

      this.setupEventHandlers();

      this.process.on('error', (error) => {
        reject(new Error(`啟動 ${this.serverName} 失敗: ${error.message}`));
      });

      // 發送初始化消息
      setTimeout(() => {
        this.initialize();
        setTimeout(() => resolve(), 3000);
      }, 1000);
    });
  }

  /**
   * 設置事件處理器
   */
  setupEventHandlers() {
    this.process.stdout.on('data', (data) => {
      this.handleStdout(data);
    });

    this.process.stderr.on('data', (data) => {
      const output = data.toString();
      if (!output.includes('INFO') && !output.includes('running on stdio')) {
        Logger.debug(`${this.serverName} stderr: ${output}`);
      }
    });

    this.process.on('close', (code) => {
      Logger.warning(`${this.serverName} 進程結束，代碼: ${code}`);
    });
  }

  /**
   * 處理標準輸出
   */
  handleStdout(data) {
    this.messageBuffer += data.toString();

    const lines = this.messageBuffer.split('\n');
    this.messageBuffer = lines.pop() || '';

    for (const line of lines) {
      if (line.trim()) {
        try {
          const message = JSON.parse(line.trim());
          this.handleMessage(message);
        } catch (error) {
          // 忽略非 JSON 行
        }
      }
    }
  }

  /**
   * 發送初始化消息
   */
  initialize() {
    this.sendMessage({
      jsonrpc: "2.0",
      id: this.requestId++,
      method: "initialize",
      params: {
        protocolVersion: "2024-11-05",
        capabilities: {
          tools: {}
        },
        clientInfo: {
          name: "crypto-predictor-client",
          version: "1.0.0"
        }
      }
    });
  }

  /**
   * 處理接收到的消息
   */
  handleMessage(message) {
    Logger.debug(`${this.serverName} 消息: ${JSON.stringify(message)}`);

    // 處理初始化響應
    if (message.id && message.result && message.result.capabilities) {
      this.handleInitializationResponse();
      return;
    }

    // 處理工具列表響應
    if (message.result && message.result.tools) {
      this.tools = message.result.tools;
      Logger.success(`${this.serverName} 工具載入完成: ${this.tools.length} 個`);
      return;
    }

    // 處理工具調用響應
    if (message.id && (message.result || message.error)) {
      this.handleToolCallResponse(message);
      return;
    }
  }

  /**
   * 處理初始化響應
   */
  handleInitializationResponse() {
    Logger.success(`${this.serverName} 初始化成功`);
    this.ready = true;

    // 發送 initialized 通知
    this.sendMessage({
      jsonrpc: "2.0",
      method: "notifications/initialized",
      params: {}
    });

    // 請求工具列表
    setTimeout(() => {
      this.requestToolsList();
    }, 500);
  }

  /**
   * 處理工具調用響應
   */
  handleToolCallResponse(message) {
    if (this.pendingRequests.has(message.id)) {
      const pendingRequest = this.pendingRequests.get(message.id);
      clearTimeout(pendingRequest.timeout);
      this.pendingRequests.delete(message.id);

      if (message.error) {
        Logger.error(`工具執行錯誤 ${pendingRequest.toolName}: ${message.error.message}`);
        pendingRequest.reject(new Error(message.error.message || '工具執行錯誤'));
      } else {
        Logger.success(`工具執行成功: ${pendingRequest.toolName}`);
        pendingRequest.resolve(message.result?.content || message.result || message);
      }
    }
  }

  /**
   * 發送消息到 MCP 服務器
   */
  sendMessage(message) {
    if (this.process && this.process.stdin.writable) {
      const jsonMessage = JSON.stringify(message) + '\n';
      Logger.debug(`發送給 ${this.serverName}: ${jsonMessage.trim()}`);
      this.process.stdin.write(jsonMessage);
    }
  }

  /**
   * 請求工具列表
   */
  requestToolsList() {
    this.sendMessage({
      jsonrpc: "2.0",
      id: this.requestId++,
      method: "tools/list",
      params: {}
    });
  }

  /**
   * 執行工具調用
   */
  async executeToolCall(toolName, arguments_) {
    return new Promise((resolve, reject) => {
      const requestId = this.requestId++;

      const timeout = setTimeout(() => {
        this.pendingRequests.delete(requestId);
        reject(new Error(`工具 ${toolName} 執行超時`));
      }, 30000);

      this.pendingRequests.set(requestId, {
        resolve,
        reject,
        timeout,
        toolName
      });

      Logger.tool(`調用工具: ${toolName}`, arguments_);

      this.sendMessage({
        jsonrpc: "2.0",
        id: requestId,
        method: "tools/call",
        params: {
          name: toolName,
          arguments: arguments_
        }
      });
    });
  }

  /**
   * 獲取工具列表
   */
  getTools() {
    return this.tools;
  }

  /**
   * 關閉連接
   */
  close() {
    if (this.process && !this.process.killed) {
      this.process.kill();
      Logger.info(`已關閉 MCP 服務器: ${this.serverName}`);
    }
  }
}
