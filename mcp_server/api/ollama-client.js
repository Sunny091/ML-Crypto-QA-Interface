// api/ollama-client.js - Ollama API 客戶端
import OpenAI from 'openai';
import { Logger } from '../utils/logger.js';
import config from '../config/environment.js';

export class OllamaClient {
  constructor() {
    this.openai = new OpenAI({
      baseURL: config.ollamaHost,
      apiKey: 'ollama', // Placeholder, actual auth via defaultHeaders
      timeout: config.ollamaTimeout,
      defaultHeaders: {
        'X-API-Key': config.ollamaApiKey,
      },
    });

    this.defaultModel = config.ollamaModel;
    this.systemPrompt = config.systemPrompt;
  }

  /**
   * 與 Ollama 對話
   */
  async chat(messages, tools = [], options = {}) {
    const openaiMessages = [
      { role: 'system', content: options.system || this.systemPrompt },
      ...this.convertMessages(messages)
    ];

    const requestConfig = {
      model: options.model || this.defaultModel,
      messages: openaiMessages,
      temperature: options.temperature || 0.7,
      max_tokens: options.max_tokens || 4096,
    };

    if (tools && tools.length > 0) {
      requestConfig.tools = tools;
    }

    try {
      Logger.api(`發送請求到 Ollama (model: ${requestConfig.model})`);
      Logger.debug('請求訊息:', JSON.stringify(openaiMessages, null, 2));

      const response = await this.openai.chat.completions.create(requestConfig);

      Logger.response('Ollama 回應:', JSON.stringify(response.choices[0].message, null, 2));

      return this.formatResponse(response);
    } catch (error) {
      Logger.error("Ollama API 調用失敗:", error.message);
      throw error;
    }
  }

  /**
   * 轉換訊息格式
   */
  convertMessages(messages) {
    const result = [];

    for (const msg of messages) {
      if (msg.role === 'user') {
        if (typeof msg.content === 'string') {
          result.push({ role: 'user', content: msg.content });
        } else if (Array.isArray(msg.content)) {
          // 處理工具結果
          for (const item of msg.content) {
            if (item.type === 'text') {
              result.push({ role: 'user', content: item.text });
            } else if (item.type === 'tool_result') {
              result.push({
                role: 'tool',
                tool_call_id: item.tool_use_id,
                content: typeof item.content === 'string'
                  ? item.content
                  : JSON.stringify(item.content)
              });
            }
          }
        }
      } else if (msg.role === 'assistant') {
        if (typeof msg.content === 'string') {
          result.push({ role: 'assistant', content: msg.content });
        } else if (Array.isArray(msg.content)) {
          const textContent = msg.content
            .filter(c => c.type === 'text')
            .map(c => c.text)
            .join('\n');

          const toolCalls = msg.content
            .filter(c => c.type === 'tool_use')
            .map(c => ({
              id: c.id,
              type: 'function',
              function: {
                name: c.name,
                arguments: JSON.stringify(c.input)
              }
            }));

          const assistantMsg = { role: 'assistant', content: textContent || null };
          if (toolCalls.length > 0) {
            assistantMsg.tool_calls = toolCalls;
          }
          result.push(assistantMsg);
        }
      }
    }

    return result;
  }

  /**
   * 格式化回應
   */
  formatResponse(response) {
    const message = response.choices[0].message;
    const content = [];

    if (message.content) {
      content.push({
        type: 'text',
        text: message.content
      });
    }

    if (message.tool_calls && message.tool_calls.length > 0) {
      for (const toolCall of message.tool_calls) {
        content.push({
          type: 'tool_use',
          id: toolCall.id,
          name: toolCall.function.name,
          input: JSON.parse(toolCall.function.arguments)
        });
      }
    }

    return {
      id: response.id,
      model: response.model,
      role: 'assistant',
      content: content,
      stop_reason: response.choices[0].finish_reason === 'tool_calls' ? 'tool_use' : 'end_turn',
      usage: {
        input_tokens: response.usage?.prompt_tokens || 0,
        output_tokens: response.usage?.completion_tokens || 0
      }
    };
  }
}
