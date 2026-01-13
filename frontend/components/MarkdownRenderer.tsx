// MarkdownRenderer.tsx
// Purpose: Reusable component to render markdown with custom link styling
// Converts markdown download links from LLM into clickable download links

import React from 'react';
import ReactMarkdown from 'react-markdown';
import './MarkdownRenderer.css';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

export function MarkdownRenderer({ content }: MarkdownRendererProps) {
  // Normalize content: remove line breaks within markdown links
  // Fix: [text]\n(url) → [text](url)
  const normalizedContent = content.replace(/\]\s*\n\s*\(/g, '](');

  return (
    <ReactMarkdown
      components={{
        // Customize link rendering for download links
        a: ({ node, ...props }) => {
          // Backend returns unified API format: /api/files/{file_id}
          // All file types (uploaded, downloaded, created) use this format
          const href = props.href || '';

          return (
            <a
              {...props}
              href={href}
              className="download-link"
              // Remove download attribute - let backend's Content-Disposition header handle it
              // download attribute only works for same-origin files
              target="_blank"
              rel="noopener noreferrer"
            >
              {props.children}
            </a>
          );
        },
      }}
    >
      {normalizedContent}
    </ReactMarkdown>
  );
}