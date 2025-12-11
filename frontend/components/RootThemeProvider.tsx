"use client";
import { ThemeProvider } from "./theme-context";

export default function RootThemeProvider({ children }: { children: React.ReactNode }) {
  return <ThemeProvider>{children}</ThemeProvider>;
}
