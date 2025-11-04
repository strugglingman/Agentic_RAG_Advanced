"use client";
import SidebarFilters from "./SidebarFilters";
import { useFilters } from "./filters-context"
import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
  { id: "chat",   label: "Chat",   href: "/chat" },
  { id: "upload", label: "Upload", href: "/upload" },
  { id: "files",  label: "Files",  href: "/files" },
  { id: "settings", label: "Settings", href: "/settings" },
];

export default function Sidebar() {
  const { selectedExts, setSelectedExts, selectedTags, setSelectedTags, customTags, setCustomTags } = useFilters();
  const pathname = usePathname();
  return (
    <div className="p-3 space-y-6">
      <div className="px-2">
        <h2 className="text-xl font-semibold tracking-tight">RAG Chatbot</h2>
      </div>
      {/* Navigation */}
      <nav className="flex flex-col gap-2" aria-label="Sidebar">
        {navItems.map(item => {
          const active = item.href === pathname || pathname.startsWith(item.href + "/");
          return (
            <Link
              key={item.id}
              href={item.href}
              aria-current={active ? "page" : undefined}
              className={
                [
                  "group w-full text-left px-4 py-2 rounded-lg text-sm font-medium transition",
                  "border",
                  active
                    ? "bg-blue-500 text-white border-blue-500 shadow-sm"
                    : "bg-white text-neutral-700 border-neutral-200 hover:border-blue-400 hover:bg-blue-50",
                  "hover:shadow-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500",
                  "active:scale-[0.98]"
                ].join(" ")
              }
            >
              <span className="flex items-center gap-2">
                <span className={
                  active
                    ? "inline-block w-2 h-2 rounded-full bg-white"
                    : "inline-block w-2 h-2 rounded-full bg-blue-300 opacity-0 group-hover:opacity-100 transition"
                } />
                {item.label}
              </span>
            </Link>
          );
        })}
      </nav>

      <hr className="border-neutral-200" />
      {/* Filters */}
      <SidebarFilters
        selectedExts={selectedExts}
        setSelectedExts={setSelectedExts}
        selectedTags={selectedTags}
        setSelectedTags={setSelectedTags}
        customTags={customTags}
        setCustomTags={setCustomTags}
      />
    </div>
  );
}