'use client';

import { Plan, ToolResult } from '@/lib/api';
import { JsonDisplay } from './JsonDisplay';
import { ToolResultDisplay } from './ToolResultDisplay';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  plan?: Plan;
  tool_called?: boolean;
  tool_name?: string;
  tool_result?: ToolResult;
  timestamp: Date;
  error?: boolean;
}

interface ChatMessageProps {
  message: Message;
}

/**
 * Single chat message component
 * Displays user messages and assistant responses with JSON plans and tool results
 */
export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`max-w-[80%] ${
          isUser
            ? 'bg-blue-600 text-white rounded-l-xl rounded-tr-xl'
            : message.error
            ? 'bg-red-50 border border-red-200 rounded-r-xl rounded-tl-xl'
            : 'bg-white border border-gray-200 rounded-r-xl rounded-tl-xl shadow-sm'
        } px-4 py-3`}
      >
        {/* Message content */}
        <p className={`${isUser ? 'text-white' : message.error ? 'text-red-700' : 'text-gray-800'} whitespace-pre-wrap`}>
          {message.content}
        </p>

        {/* Tool Result display (Phase 3) */}
        {!isUser && message.tool_called && message.tool_result && (
          <div className="mt-3">
            <ToolResultDisplay
              tool_name={message.tool_name || 'unknown'}
              tool_result={message.tool_result}
            />
          </div>
        )}

        {/* JSON Plan display for assistant messages */}
        {!isUser && message.plan && (
          <div className="mt-3">
            <p className="text-xs text-gray-500 mb-2 font-medium">JSON Plan:</p>
            <JsonDisplay plan={message.plan} />
          </div>
        )}

        {/* Timestamp */}
        <p
          className={`text-xs mt-2 ${
            isUser ? 'text-blue-200' : 'text-gray-400'
          }`}
        >
          {message.timestamp.toLocaleTimeString('zh-TW', {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </p>
      </div>
    </div>
  );
}
