import "./globals.css";
import RootThemeProvider from "@/components/RootThemeProvider";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="h-screen overflow-hidden bg-neutral-100 dark:bg-neutral-900">
        <RootThemeProvider>
          {children}
        </RootThemeProvider>
      </body>
    </html>
  );
}