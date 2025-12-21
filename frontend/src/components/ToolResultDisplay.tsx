'use client';

import { ToolResult } from '@/lib/api';

interface ToolResultDisplayProps {
  tool_name: string;
  tool_result: ToolResult;
}

/**
 * Display component for tool execution results
 * Shows prediction, probability, and model info
 */
export function ToolResultDisplay({ tool_name, tool_result }: ToolResultDisplayProps) {
  const isUp = tool_result.prediction === 'UP';
  const confidence = (tool_result.prob_up * 100).toFixed(1);

  return (
    <div className="bg-gradient-to-r from-green-50 to-blue-50 border border-green-200 rounded-lg p-4">
      {/* Tool info header */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-green-600 text-lg">&#9989;</span>
        <span className="text-sm font-medium text-gray-600">
          Tool: <code className="bg-gray-100 px-1 rounded">{tool_name}</code>
        </span>
      </div>

      {/* Prediction result */}
      <div className="flex items-center gap-4 mb-3 flex-wrap">
        <div className="text-center">
          <div className="text-xs text-gray-500 mb-1">{tool_result.symbol}</div>
          <div
            className={`text-2xl font-bold ${
              isUp ? 'text-green-600' : 'text-red-600'
            }`}
          >
            {isUp ? '&#9650; UP' : '&#9660; DOWN'}
          </div>
        </div>
        <div className="text-center">
          <div className="text-xs text-gray-500 mb-1">Confidence</div>
          <div className="text-xl font-semibold text-gray-700">{confidence}%</div>
        </div>
        <div className="text-center">
          <div className="text-xs text-gray-500 mb-1">As of</div>
          <div className="text-sm font-medium text-gray-700">{tool_result.as_of}</div>
        </div>
        {/* Phase 4: Sentiment score display */}
        {tool_result.sentiment_score !== undefined && (
          <div className="text-center">
            <div className="text-xs text-gray-500 mb-1">Sentiment</div>
            <div
              className={`text-lg font-semibold ${
                tool_result.sentiment_score > 0.2
                  ? 'text-green-600'
                  : tool_result.sentiment_score < -0.2
                  ? 'text-red-600'
                  : 'text-gray-600'
              }`}
            >
              {tool_result.sentiment_score > 0.2
                ? '😊 Positive'
                : tool_result.sentiment_score < -0.2
                ? '😟 Negative'
                : '😐 Neutral'}
            </div>
            <div className="text-xs text-gray-400">
              ({tool_result.sentiment_score.toFixed(2)})
            </div>
          </div>
        )}
      </div>

      {/* Model info */}
      <div className="text-xs text-gray-500 border-t border-gray-200 pt-2 mt-2">
        <span className="font-medium">Model:</span> {tool_result.model_variant}
        <span className="mx-2">|</span>
        <span className="font-medium">Features:</span> {tool_result.features_used.join(', ')}
      </div>

      {/* Raw JSON (collapsible) */}
      <details className="mt-3">
        <summary className="text-xs text-gray-400 cursor-pointer hover:text-gray-600">
          View raw JSON
        </summary>
        <pre className="mt-2 bg-gray-900 text-gray-100 text-xs p-3 rounded overflow-x-auto">
          {JSON.stringify(tool_result, null, 2)}
        </pre>
      </details>
    </div>
  );
}
