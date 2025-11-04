"use client";
import React, { createContext, useContext, useState } from "react";

type FiltersCtx = {
  selectedExts: Set<string>;
  setSelectedExts: React.Dispatch<React.SetStateAction<Set<string>>>;
  selectedTags: string[];
  setSelectedTags: React.Dispatch<React.SetStateAction<string[]>>;
  customTags: string;
  setCustomTags: React.Dispatch<React.SetStateAction<string>>;
};

const Ctx = createContext<FiltersCtx | null>(null);
export const useFilters = () => {
  const v = useContext(Ctx);
  if (!v) throw new Error("useFilters must be used within <FiltersProvider>");
  return v;
};

export function FiltersProvider({ children }: { children: React.ReactNode }) {
  const [selectedExts, setSelectedExts] = useState<Set<string>>(new Set());
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [customTags, setCustomTags] = useState("");

  return (
    <Ctx.Provider value={{ selectedExts, setSelectedExts, selectedTags, setSelectedTags, customTags, setCustomTags }}>
      {children}
    </Ctx.Provider>
  );
}
