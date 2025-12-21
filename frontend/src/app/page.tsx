'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { ChatMessage, Message } from '@/components/ChatMessage';
import { ChatInput } from '@/components/ChatInput';
import { DebugPanel, LogEntry } from '@/components/DebugPanel';
import { sendChatMessage, checkHealth } from '@/lib/api';

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'checking' | 'connected' | 'error'>('checking');
  const [modelName, setModelName] = useState<string>('');
  const [debugLogs, setDebugLogs] = useState<LogEntry[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Add a log entry
  const addLog = useCallback((type: LogEntry['type'], title: string, details?: string | object) => {
    const entry: LogEntry = {
      id: `log-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
      timestamp: new Date(),
      type,
      title,
      details,
    };
    setDebugLogs(prev => [...prev, entry]);
  }, []);

  // Clear logs
  const clearLogs = useCallback(() => {
    setDebugLogs([]);
  }, []);

  // Check backend health on mount
  useEffect(() => {
    const checkConnection = async () => {
      addLog('info', '正在連線 Ollama...');
      try {
        const health = await checkHealth();
        setConnectionStatus('connected');
        setModelName(health.model);
        addLog('info', `Ollama 已連線`, { model: health.model, env: health.env });
      } catch (error) {
        setConnectionStatus('error');
        addLog('error', 'Ollama 連線失敗', error instanceof Error ? error.message : String(error));
      }
    };
    checkConnection();
  }, [addLog]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async (content: string) => {
    // Add user message
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    // Log request
    addLog('request', '發送請求', { message: content });

    try {
      const response = await sendChatMessage(content);

      // Log response with timing
      const timing = response._timing;
      addLog('response', `收到回應 (${timing?.durationMs || 0}ms)`, {
        plan: response.plan,
        tool_called: response.tool_called,
      });

      // Log tool call if any
      if (response.tool_called && response.tool_name) {
        addLog('tool', `呼叫工具: ${response.tool_name}`, response.tool_result);
      }

      // Add assistant message with plan and tool result (Phase 3)
      const assistantMessage: Message = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: response.assistant_text,
        plan: response.plan,
        tool_called: response.tool_called,
        tool_name: response.tool_name,
        tool_result: response.tool_result,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      // Log error
      addLog('error', '請求失敗', error instanceof Error ? error.message : String(error));

      // Add error message
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error occurred'}`,
        timestamp: new Date(),
        error: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-4 py-3 shadow-sm">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold text-gray-800">
              MCP Orchestrator
            </h1>
            <p className="text-sm text-gray-500">Phase 3: Full Integration</p>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={`inline-block w-2 h-2 rounded-full ${
                connectionStatus === 'connected'
                  ? 'bg-green-500'
                  : connectionStatus === 'error'
                  ? 'bg-red-500'
                  : 'bg-yellow-500 animate-pulse'
              }`}
            />
            <span className="text-sm text-gray-600">
              {connectionStatus === 'connected'
                ? `Connected (${modelName})`
                : connectionStatus === 'error'
                ? 'Disconnected'
                : 'Connecting...'}
            </span>
          </div>
        </div>
      </header>

      {/* Messages area */}
      <main className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-4xl mx-auto">
          {messages.length === 0 ? (
            <div className="text-center py-12">
              <div className="text-6xl mb-4">&#128640;</div>
              <h2 className="text-xl font-medium text-gray-700 mb-2">
                Welcome to MCP Orchestrator
              </h2>
              <p className="text-gray-500 mb-6">
                Phase 3: Natural language to prediction via MCP tools
              </p>
              <div className="bg-gray-100 rounded-lg p-4 text-left max-w-md mx-auto">
                <p className="text-sm font-medium text-gray-700 mb-2">
                  Try these examples:
                </p>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>&#8226; 幫我預測 BTC 現在的隔日漲跌</li>
                  <li>&#8226; 預測 2023-06-01 的 ETH 明天走勢</li>
                  <li>&#8226; SOL 明天會漲還是跌？</li>
                </ul>
                <p className="text-xs text-gray-400 mt-3">
                  Note: 新聞分析功能尚未開放 (Phase 4)
                </p>
              </div>
            </div>
          ) : (
            messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Debug Panel */}
      <div className="px-4 pb-2">
        <div className="max-w-4xl mx-auto">
          <DebugPanel
            logs={debugLogs}
            ollamaStatus={connectionStatus}
            ollamaModel={modelName}
            onClear={clearLogs}
          />
        </div>
      </div>

      {/* Input area */}
      <ChatInput onSend={handleSend} disabled={isLoading} />
    </div>
  );
}
