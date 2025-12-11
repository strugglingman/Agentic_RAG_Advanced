# Frontend Conversation History - Implementation Guide

## Overview
Transform the chatbot UI to support persistent conversation history with a modern ChatGPT-like interface.

---

## 1. Architecture Changes

### Current State
- Chat history stored in `localStorage`
- No conversation management
- Single continuous chat session

### Target State
- Conversations stored in PostgreSQL (via backend API)
- Sidebar with conversation list
- Create/switch/delete conversations
- Persistent across devices

---

## 2. API Integration

### Create API Proxy Routes

**File: `frontend/src/app/api/conversations/route.ts`**

```typescript
import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:5001';

export async function GET(request: NextRequest) {
  const token = request.headers.get('Authorization');

  const response = await fetch(`${BACKEND_URL}/conversations`, {
    headers: { 'Authorization': token || '' }
  });

  return NextResponse.json(await response.json());
}
```

**File: `frontend/src/app/api/conversations/[id]/route.ts`**

```typescript
export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  const token = request.headers.get('Authorization');

  const response = await fetch(`${BACKEND_URL}/conversations/${params.id}`, {
    headers: { 'Authorization': token || '' }
  });

  return NextResponse.json(await response.json());
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  const token = request.headers.get('Authorization');

  const response = await fetch(`${BACKEND_URL}/conversations/${params.id}`, {
    method: 'DELETE',
    headers: { 'Authorization': token || '' }
  });

  return NextResponse.json(await response.json());
}
```

---

## 3. State Management

### Remove localStorage, Use React State + API

**File: `frontend/src/contexts/chat-context.tsx`**

```typescript
interface ChatContextType {
  conversations: Conversation[];
  currentConversationId: string | null;
  messages: Message[];
  isLoading: boolean;

  // Actions
  loadConversations: () => Promise<void>;
  createNewConversation: () => void;
  switchConversation: (id: string) => Promise<void>;
  deleteConversation: (id: string) => Promise<void>;
  sendMessage: (content: string) => Promise<void>;
}

interface Conversation {
  id: string;
  title: string;
  updated_at: string;
  preview: string;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
}
```

**Key Changes:**
- Replace `localStorage.getItem('chatHistory')` with API calls
- Add `currentConversationId` to track active conversation
- `sendMessage()` includes `conversation_id` in request

---

## 4. UI Components

### A. Conversation Sidebar (New Component)

**File: `frontend/src/components/ConversationSidebar.tsx`**

**Design:**
```
┌─────────────────────────┐
│  [+ New Chat]           │
│─────────────────────────│
│ ● Weather in Kungälv    │ ← Active
│   "The weather tomor..." │
│   2 hours ago           │
├─────────────────────────┤
│   Document Search       │
│   "I found 3 docume..." │
│   Yesterday             │
├─────────────────────────┤
│   Calculate Revenue     │
│   "The total is..."     │
│   2 days ago            │
└─────────────────────────┘
```

**Features:**
- New Chat button at top
- List sorted by `updated_at` (newest first)
- Active conversation highlighted
- Preview of last message (50 chars)
- Hover: Show delete button
- Click: Switch conversation
- Responsive: Collapsible on mobile

**Styling (Tailwind):**
```tsx
<div className="flex flex-col h-full bg-gray-900 text-white">
  {/* New Chat Button */}
  <button className="m-4 p-3 border border-gray-600 rounded-lg hover:bg-gray-800">
    + New Chat
  </button>

  {/* Conversation List */}
  <div className="flex-1 overflow-y-auto">
    {conversations.map(conv => (
      <div
        key={conv.id}
        className={`p-4 cursor-pointer hover:bg-gray-800 ${
          conv.id === currentId ? 'bg-gray-800 border-l-4 border-blue-500' : ''
        }`}
      >
        <h3 className="font-semibold truncate">{conv.title}</h3>
        <p className="text-sm text-gray-400 truncate">{conv.preview}</p>
        <span className="text-xs text-gray-500">{formatTime(conv.updated_at)}</span>
      </div>
    ))}
  </div>
</div>
```

---

### B. Main Chat Layout Update

**File: `frontend/src/app/chat/page.tsx`**

```
┌──────────────────────────────────────────────────┐
│  [≡] Sidebar    RAG Chatbot                      │
├──────────┬───────────────────────────────────────┤
│          │  User: What's the weather?            │
│ Sidebar  │  Bot: The weather in Kungälv is...    │
│          │                                        │
│ (250px)  │  [Input box]                    [Send]│
└──────────┴───────────────────────────────────────┘
```

**Layout:**
```tsx
<div className="flex h-screen">
  {/* Sidebar */}
  <ConversationSidebar
    conversations={conversations}
    currentId={currentConversationId}
    onNew={createNewConversation}
    onSwitch={switchConversation}
    onDelete={deleteConversation}
  />

  {/* Main Chat */}
  <div className="flex-1 flex flex-col">
    <ChatMessages messages={messages} />
    <ChatInput onSend={sendMessage} />
  </div>
</div>
```

---

### C. Chat Input Update

**File: `frontend/src/components/ChatInput.tsx`**

**Change:** Include `conversation_id` when sending messages

```typescript
const handleSend = async (message: string) => {
  const response = await fetch('/api/chat/agent', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      conversation_id: currentConversationId, // ← NEW
      messages: [{ role: 'user', content: message }]
    })
  });
};
```

---

## 5. User Flows

### Flow 1: New Chat
1. User clicks "+ New Chat"
2. Frontend sets `currentConversationId = null`
3. Clear `messages = []`
4. User sends first message
5. Backend creates conversation automatically (lazy creation)
6. Backend returns new `conversation_id` in response
7. Frontend updates `currentConversationId`
8. Reload conversation list

### Flow 2: Switch Conversation
1. User clicks conversation in sidebar
2. Call `GET /api/conversations/{id}`
3. Load messages into chat view
4. Set `currentConversationId = id`

### Flow 3: Delete Conversation
1. User hovers over conversation → shows delete icon
2. Click delete → confirm dialog
3. Call `DELETE /api/conversations/{id}`
4. Remove from sidebar list
5. If deleted current conversation → switch to "New Chat"

---

## 6. Modern UI Features

### A. Smooth Transitions
```css
.conversation-item {
  transition: background-color 0.2s ease;
}

.message-appear {
  animation: slideUp 0.3s ease-out;
}

@keyframes slideUp {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
```

### B. Loading States
```tsx
{isLoading && (
  <div className="flex items-center gap-2">
    <div className="animate-spin h-4 w-4 border-2 border-blue-500" />
    <span>Loading conversations...</span>
  </div>
)}
```

### C. Empty States
```tsx
{conversations.length === 0 && (
  <div className="text-center p-8 text-gray-500">
    <p>No conversations yet</p>
    <p className="text-sm">Start a new chat to begin</p>
  </div>
)}
```

### D. Responsive Design
```tsx
// Mobile: Hide sidebar by default, show with hamburger menu
<div className="hidden md:block md:w-64">
  <ConversationSidebar />
</div>

// Mobile menu button
<button className="md:hidden" onClick={toggleSidebar}>
  ☰
</button>
```

---

## 7. Implementation Checklist

### Phase 1: Backend Integration
- [ ] Create `/api/conversations` proxy routes
- [ ] Create `/api/conversations/[id]` proxy routes
- [ ] Test API calls with Postman/curl

### Phase 2: State Management
- [ ] Remove `localStorage` from chat-context.tsx
- [ ] Add `conversations` state
- [ ] Add `currentConversationId` state
- [ ] Implement `loadConversations()`
- [ ] Implement `switchConversation(id)`
- [ ] Implement `deleteConversation(id)`
- [ ] Update `sendMessage()` to include `conversation_id`

### Phase 3: UI Components
- [ ] Create `ConversationSidebar.tsx`
- [ ] Update chat layout with sidebar
- [ ] Add "New Chat" button
- [ ] Add conversation list rendering
- [ ] Add delete confirmation dialog
- [ ] Add loading/empty states

### Phase 4: Polish
- [ ] Add animations/transitions
- [ ] Mobile responsive design
- [ ] Error handling (network failures)
- [ ] Optimistic UI updates
- [ ] Auto-scroll to latest message
- [ ] Format timestamps (e.g., "2 hours ago")

---

## 8. Key Technical Notes

### A. Lazy Conversation Creation
- Don't create conversation on "New Chat" click
- Create when user sends first message
- Backend handles this automatically

### B. Token Management
```typescript
// Get token from auth context
const { token } = useAuth();

// Include in all API calls
headers: { 'Authorization': `Bearer ${token}` }
```

### C. Error Handling
```typescript
try {
  const res = await fetch('/api/conversations');
  if (!res.ok) throw new Error('Failed to load');
  const data = await res.json();
  setConversations(data.conversations);
} catch (error) {
  console.error(error);
  // Show error toast/message
}
```

### D. Real-time Updates
- After sending message, reload conversation list (updated timestamps)
- After deleting, remove from local state immediately
- Use optimistic updates for better UX

---

## 9. Color Scheme (Modern Dark Theme)

```css
:root {
  --bg-primary: #0f172a;      /* Main background */
  --bg-secondary: #1e293b;    /* Sidebar background */
  --bg-hover: #334155;        /* Hover state */
  --bg-active: #475569;       /* Active conversation */
  --text-primary: #f8fafc;    /* Main text */
  --text-secondary: #94a3b8;  /* Secondary text */
  --border: #334155;          /* Borders */
  --accent: #3b82f6;          /* Active indicator */
}
```

---

## 10. Files to Modify/Create

### Create:
- `frontend/src/app/api/conversations/route.ts`
- `frontend/src/app/api/conversations/[id]/route.ts`
- `frontend/src/components/ConversationSidebar.tsx`
- `frontend/src/components/DeleteConfirmDialog.tsx` (optional)

### Modify:
- `frontend/src/contexts/chat-context.tsx` (remove localStorage, add API)
- `frontend/src/app/chat/page.tsx` (add sidebar)
- `frontend/src/components/ChatInput.tsx` (add conversation_id)
- `frontend/src/types/index.ts` (add Conversation type)

---

## Done!

This guide provides everything needed to implement a modern, ChatGPT-like conversation interface. Focus on getting core functionality working first, then polish the UI.
