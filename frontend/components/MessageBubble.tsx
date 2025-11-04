export default function MessageBubble({ role, content }: { role: "user" | "assistant"; content: string }) {
    const isUser = role === "user";
    return (
        <div className={`my-2 flex ${isUser ? "justify-end" : "justify-start"}`}>
            <div className={`max-w-[80%] rounded-2xl px-3 py-2 text-sm whitespace-pre-wrap ${isUser ? "bg-blue-100" : "bg-white border"}`}>
                {content}
            </div>
        </div>
    );
}