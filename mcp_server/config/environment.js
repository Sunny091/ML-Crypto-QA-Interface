// config/environment.js - 環境配置管理
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({ path: path.join(__dirname, '..', '.env') });

class EnvironmentConfig {
  constructor() {
    this.mcpConfig = this.loadMCPConfig();
    this.validateEnvironment();
  }

  validateEnvironment() {
    if (!this.ollamaHost) {
      console.error('[ERROR] OLLAMA_HOST 未設置');
      process.exit(1);
    }

    console.log('[CONFIG] 環境配置:');
    console.log(`  - Ollama: ${this.ollamaHost}`);
    console.log(`  - Model: ${this.ollamaModel}`);
    console.log(`  - Tools Path: ${this.toolsPath}`);
  }

  loadMCPConfig() {
    try {
      const configPath = path.join(__dirname, '..', 'config.json');
      const config = JSON.parse(fs.readFileSync(configPath, 'utf-8'));

      // 替換環境變數
      if (config.mcpServers) {
        Object.keys(config.mcpServers).forEach(serverName => {
          const server = config.mcpServers[serverName];
          if (server.args && Array.isArray(server.args)) {
            server.args = server.args.map(arg =>
              this.replaceEnvironmentVariables(arg)
            );
          }
        });
      }

      return config;
    } catch (error) {
      console.error('[ERROR] 無法讀取 config.json:', error.message);
      process.exit(1);
    }
  }

  replaceEnvironmentVariables(str) {
    return str.replace(/\$\{([^}]+)\}/g, (match, envVar) => {
      return process.env[envVar] || match;
    });
  }

  get ollamaHost() {
    return process.env.OLLAMA_HOST;
  }

  get ollamaModel() {
    return process.env.OLLAMA_MODEL || 'gpt-oss:20b';
  }

  get ollamaApiKey() {
    return process.env.OLLAMA_API_KEY || 'ollama';
  }

  get ollamaTimeout() {
    return parseInt(process.env.OLLAMA_TIMEOUT_MS) || 60000;
  }

  get toolsPath() {
    return process.env.TOOLS_PATH || path.join(__dirname, '..', 'tools');
  }

  get mcpServers() {
    return this.mcpConfig.mcpServers || {};
  }

  get systemPrompt() {
    return this.mcpConfig.systemPrompt || '你是一個加密貨幣價格預測助手。';
  }

  get port() {
    return parseInt(process.env.PORT) || 3000;
  }

  get host() {
    return process.env.HOST || '0.0.0.0';
  }

  get appEnv() {
    return process.env.APP_ENV || 'development';
  }
}

export default new EnvironmentConfig();
