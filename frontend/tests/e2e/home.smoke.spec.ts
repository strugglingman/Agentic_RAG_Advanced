import { test, expect } from "@playwright/test";

test("home page renders", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Agentic RAG Chatbot/i })).toBeVisible();
});
