'use client';

import { useState } from 'react';

export interface LogEntry {
  id: string;
  timestamp: Date;
  type: 'info' | 'request' | 'response' | 'tool' | 'error';
  title: string;
  details?: string | object;
}

interface DebugPanelProps {
  logs: LogEntry[];
  ollamaStatus: 'checking' | 'connected' | 'error';
  ollamaModel: string;
  onClear: () => void;
}

/**
 * Debug panel component for displaying execution logs
 */
export function DebugPanel({ logs, ollamaStatus, ollamaModel, onClear }: DebugPanelProps) {
  const [isExpanded, setIsExpanded] = useState(true);
  const [expandedLogs, setExpandedLogs] = useState<Set<string>>(new Set());

  const toggleLog = (id: string) => {
    setExpandedLogs(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const getTypeIcon = (type: LogEntry['type']) => {
    switch (type) {
      case 'info': return '📋';
      case 'request': return '📤';
      case 'response': return '📥';
      case 'tool': return '🔧';
      case 'error': return '❌';
    }
  };

  const getTypeColor = (type: LogEntry['type']) => {
    switch (type) {
      case 'info': return 'text-blue-600';
      case 'request': return 'text-purple-600';
      case 'response': return 'text-green-600';
      case 'tool': return 'text-orange-600';
      case 'error': return 'text-red-600';
    }
  };

  return (
    <div className="bg-gray-900 text-gray-100 rounded-lg shadow-lg overflow-hidden">
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-2 bg-gray-800 cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium">🔍 Debug Panel</span>
          <div className="flex items-center gap-2">
            <span
              className={`inline-block w-2 h-2 rounded-full ${
                ollamaStatus === 'connected'
                  ? 'bg-green-500'
                  : ollamaStatus === 'error'
                  ? 'bg-red-500'
                  : 'bg-yellow-500 animate-pulse'
              }`}
            />
            <span className="text-xs text-gray-400">
              Ollama: {ollamaStatus === 'connected' ? `✓ ${ollamaModel}` : ollamaStatus === 'error' ? '✗ 離線' : '連線中...'}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={(e) => { e.stopPropagation(); onClear(); }}
            className="text-xs text-gray-400 hover:text-white px-2 py-1 rounded hover:bg-gray-700"
          >
            清除
          </button>
          <span className="text-gray-500">{isExpanded ? '▼' : '▶'}</span>
        </div>
      </div>

      {/* Logs */}
      {isExpanded && (
        <div className="max-h-64 overflow-y-auto p-2 space-y-1">
          {logs.length === 0 ? (
            <div className="text-center text-gray-500 py-4 text-sm">
              尚無執行記錄
            </div>
          ) : (
            logs.map((log) => (
              <div
                key={log.id}
                className="bg-gray-800 rounded px-3 py-2 text-sm"
              >
                <div
                  className="flex items-center gap-2 cursor-pointer"
                  onClick={() => log.details && toggleLog(log.id)}
                >
                  <span>{getTypeIcon(log.type)}</span>
                  <span className="text-gray-500 text-xs">
                    {log.timestamp.toLocaleTimeString('zh-TW', {
                      hour: '2-digit',
                      minute: '2-digit',
                      second: '2-digit',
                    })}
                  </span>
                  <span className={`font-medium ${getTypeColor(log.type)}`}>
                    {log.title}
                  </span>
                  {log.details && (
                    <span className="text-gray-500 text-xs ml-auto">
                      {expandedLogs.has(log.id) ? '▼' : '▶'}
                    </span>
                  )}
                </div>
                {log.details && expandedLogs.has(log.id) && (
                  <pre className="mt-2 text-xs text-gray-400 overflow-x-auto bg-gray-900 p-2 rounded">
                    {typeof log.details === 'string'
                      ? log.details
                      : JSON.stringify(log.details, null, 2)}
                  </pre>
                )}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
