'use client';

import { Plan } from '@/lib/api';

interface JsonDisplayProps {
  plan: Plan;
}

/**
 * Formatted JSON display component
 * Shows the plan in a readable, syntax-highlighted format
 */
export function JsonDisplay({ plan }: JsonDisplayProps) {
  const renderValue = (value: unknown, key: string): React.ReactNode => {
    if (value === null) {
      return <span className="text-gray-500">null</span>;
    }
    if (typeof value === 'boolean') {
      return <span className="text-red-600">{value.toString()}</span>;
    }
    if (typeof value === 'string') {
      // Truncate long strings (like news_text)
      const displayValue = value.length > 100 ? value.slice(0, 100) + '...' : value;
      return <span className="text-green-700">&quot;{displayValue}&quot;</span>;
    }
    return <span className="text-blue-600">{String(value)}</span>;
  };

  const entries = Object.entries(plan);

  return (
    <div className="bg-gray-900 rounded-lg p-4 font-mono text-sm overflow-x-auto">
      <div className="text-gray-300">{'{'}</div>
      {entries.map(([key, value], index) => (
        <div key={key} className="pl-4">
          <span className="text-purple-400">&quot;{key}&quot;</span>
          <span className="text-gray-300">: </span>
          {renderValue(value, key)}
          {index < entries.length - 1 && <span className="text-gray-300">,</span>}
        </div>
      ))}
      <div className="text-gray-300">{'}'}</div>
    </div>
  );
}
